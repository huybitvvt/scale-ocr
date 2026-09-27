"""Merge visually audited OCR proposals into the local annotation file.

Every imported row remains a draft. Existing human annotations take priority.
Audit decisions and images belong in ignored data/, never in the public repo.
"""

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

from scale_ocr.annotations import load_annotations, save_annotations, validate_annotation


def choose_box(candidate: dict, value: str, manual: list[dict] | None) -> list[dict]:
    if manual:
        return [{**box, "text": value} for box in manual]
    options = []
    for view in ("scene", "zoom"):
        box = candidate[f"{view}_box"]
        ocr = candidate[f"{view}_text"]
        if box is None or not ocr:
            continue
        similarity = SequenceMatcher(None, ocr.replace(".", ""), value.replace(".", "")).ratio()
        score = (ocr == value, similarity, candidate[f"{view}_conf"], view == "zoom")
        options.append((score, {"xyxy": box, "view_kind": view, "text": value}))
    if not options:
        raise ValueError(f"No display box for pilot image {candidate['index']}")
    return [max(options, key=lambda item: item[0])[1]]


def build_drafts(pilot: list[dict], candidates: list[dict], audit: dict,
                 manual_boxes: dict) -> dict[str, dict]:
    if len(pilot) != len(candidates):
        raise ValueError("Pilot and candidate counts differ")
    drafts = {}
    overrides = audit["overrides"]
    statuses = audit["statuses"]
    for index, (row, candidate) in enumerate(zip(pilot, candidates), 1):
        if candidate["index"] != index or candidate["asset_id"] != row["asset_id"]:
            raise ValueError(f"Pilot/candidate mismatch at {index}")
        key = str(index)
        value = overrides[key] if key in overrides else candidate["chosen"]
        status = statuses.get(key, "readable")
        if status == "readable":
            if not value:
                raise ValueError(f"Missing visible number for pilot image {index}")
            boxes = choose_box(candidate, value, manual_boxes.get(key))
        else:
            if value is not None:
                raise ValueError(f"Non-readable pilot image {index} must have null override")
            boxes = []
        draft = {
            "asset_id": row["asset_id"],
            "readability": status,
            "review_status": "draft",
            "annotator": "codex-ocr-assisted",
            "reviewer": "",
            "reason": "OCR proposal checked visually; independent review required",
            "boxes": boxes,
        }
        drafts[row["asset_id"]] = validate_annotation(draft, row)
    return drafts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manual-boxes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.pilot.open(encoding="utf-8", newline="") as stream:
        pilot = list(csv.DictReader(stream))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manual_boxes = json.loads(args.manual_boxes.read_text(encoding="utf-8"))
    drafts = build_drafts(pilot, candidates, audit, manual_boxes)
    existing = load_annotations(args.out)
    unknown = set(existing) - set(drafts)
    if unknown:
        raise ValueError(f"Existing annotations outside pilot: {len(unknown)}")
    merged = {**drafts, **existing}
    save_annotations(args.out, merged)
    from collections import Counter
    counts = Counter(item["readability"] for item in merged.values())
    print(f"Saved {len(merged)} annotations; preserved {len(existing)} existing")
    print(dict(counts))
    print("All imported labels are drafts and require independent review.")


if __name__ == "__main__":
    main()
