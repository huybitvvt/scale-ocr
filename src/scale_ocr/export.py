"""Export reviewed labels to YOLO detection and OCR recognition datasets."""

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .annotations import load_annotations, validate_annotation
from .inventory import read_csv, sha256_file, write_csv


def export_dataset(raw_root: Path, index_path: Path, split_path: Path,
                   annotations_path: Path, destination: Path, margin=0.05) -> dict:
    if not 0 <= margin <= 0.25:
        raise ValueError("margin must be between 0 and 0.25")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    index = {row["asset_id"]: row for row in read_csv(index_path)}
    splits = {row["asset_id"]: row for row in read_csv(split_path)}
    annotations = load_annotations(annotations_path)
    if set(index) != set(splits):
        raise ValueError("Index and split asset IDs differ")
    if set(annotations) - set(index):
        raise ValueError("Unknown assets in annotations")
    detector_root = destination / "detector"
    recognizer_root = destination / "recognizer"
    for split in ("train", "val", "test"):
        for base in (detector_root / "images", detector_root / "labels",
                     recognizer_root / "crops"):
            (base / split).mkdir(parents=True, exist_ok=True)
    rec_labels = {split: [] for split in ("train", "val", "test")}
    crop_map = []
    evaluation_rows = []
    counts = Counter()
    for asset_id in sorted(annotations):
        row = index[asset_id]
        split_row = splits[asset_id]
        split = split_row["split"]
        if split not in rec_labels:
            counts["excluded_annotations"] += 1
            continue
        ann = validate_annotation(annotations[asset_id], row)
        if ann["review_status"] != "reviewed":
            counts["draft_skipped"] += 1
            continue
        if split in {"val", "test"} and (
            not ann["reviewer"] or ann["reviewer"] == ann["annotator"]
        ):
            raise ValueError(f"Independent reviewer required for {split}: {asset_id}")
        source = raw_root / row["local_file"]
        if not source.is_file():
            raise FileNotFoundError(source)
        image = Image.open(source)
        image.load()
        width, height = image.size
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Image size differs from manifest: {asset_id}")
        suffix = source.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            counts["unsupported_format_skipped"] += 1
            continue
        skip_reason = ""
        if ann["readability"] not in {"readable", "partial", "unreadable", "no_display"}:
            skip_reason = "non_numeric"
        elif ann["readability"] != "no_display" and not ann["boxes"]:
            skip_reason = "no_box"
        if skip_reason:
            relative_image = Path("evaluation") / "other" / split / f"{asset_id}{suffix}"
        else:
            relative_image = Path("detector") / "images" / split / f"{asset_id}{suffix}"
        target_image = destination / relative_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_image)
        evaluation_rows.append({
            "sample_id": asset_id, "gateway": row["gateway"],
            "event_key": row["event_key"], "split": split,
            "readability": ann["readability"],
            "gt_text": ann["boxes"][0]["text"] if ann["readability"] == "readable" else "",
            "image_path": relative_image.as_posix(),
            "training_export": "no" if skip_reason else "yes",
            "skip_reason": skip_reason,
        })
        counts[f"evaluation_images_{split}"] += 1
        if skip_reason:
            counts[f"{skip_reason}_skipped"] += 1
            continue
        yolo_lines = []
        for box_index, box in enumerate(ann["boxes"]):
            x1, y1, x2, y2 = box["xyxy"]
            cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            yolo_lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            if ann["readability"] != "readable":
                continue
            padx, pady = round((x2 - x1) * margin), round((y2 - y1) * margin)
            crop = image.crop((max(0, x1 - padx), max(0, y1 - pady),
                               min(width, x2 + padx), min(height, y2 + pady)))
            name = f"{asset_id}_{box_index:02d}_{box['view_kind']}.png"
            relative = Path("crops") / split / name
            crop.save(recognizer_root / relative)
            rec_labels[split].append(f"{relative.as_posix()}\t{box['text']}\n")
            crop_map.append({
                "asset_id": asset_id, "event_key": row["event_key"],
                "group_id": split_row["group_id"], "split": split,
                "view_kind": box["view_kind"], "crop_path": relative.as_posix(),
                "display_text": box["text"],
            })
            counts[f"recognizer_crops_{split}"] += 1
        (detector_root / "labels" / split / f"{asset_id}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )
        counts[f"detector_images_{split}"] += 1
        if ann["readability"] == "no_display":
            counts[f"negative_images_{split}"] += 1
    for split, lines in rec_labels.items():
        (recognizer_root / f"{split}.txt").write_text("".join(lines), encoding="utf-8")
    write_csv(destination / "crop_map.csv", crop_map,
              ("asset_id", "event_key", "group_id", "split", "view_kind", "crop_path", "display_text"))
    write_csv(destination / "evaluation_manifest.csv", evaluation_rows,
              ("sample_id", "gateway", "event_key", "split", "readability", "gt_text",
               "image_path", "training_export", "skip_reason"))
    yaml_path = detector_root.as_posix()
    (detector_root / "data.yaml").write_text(
        f"path: {json.dumps(yaml_path)}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: weight_display\n", encoding="utf-8"
    )
    card = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256_file(raw_root / "manifest.json"),
        "index_sha256": sha256_file(index_path),
        "split_sha256": sha256_file(split_path),
        "annotations_sha256": sha256_file(annotations_path),
        "margin": margin,
        "counts": dict(sorted(counts.items())),
    }
    (destination / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return card
