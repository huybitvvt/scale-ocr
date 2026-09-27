import csv
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from scale_ocr.annotations import validate_annotation
from scale_ocr.evaluate import summarize
from scale_ocr.export import export_dataset
from scale_ocr.inventory import build_index, read_csv, write_csv
from scale_ocr.package import package_dataset
from scale_ocr.pilot import choose_pilot
from scale_ocr.splits import make_split


def fixture_backup(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    assets = []
    cases = [
        ("a", "2026/09/24", "core-weight", "event-a", (200, 0, 0)),
        ("b", "2026/09/25", "product-weight", "event-a", (0, 200, 0)),
        ("c", "2026/09/26", "core-weight", "event-b", (200, 0, 0)),
        ("d", "2026/09/27", "product-weight", "event-c", (0, 0, 200)),
    ]
    for asset_id, day, kind, event_id, color in cases:
        relative = f"assets/{asset_id}.png"
        target = raw / relative
        target.parent.mkdir(exist_ok=True)
        Image.new("RGB", (100, 60), color).save(target)
        assets.append({
            "asset_id": asset_id, "public_id": f"roll-captures/gateway-01/{day}/{kind}/{event_id}",
            "local_file": relative, "width": 100, "height": 60,
            "bytes": target.stat().st_size, "uploaded_at": "",
        })
    (raw / "manifest.json").write_text(json.dumps({"assets": assets}), encoding="utf-8")
    return raw


def test_index_split_and_pilot_prevent_event_and_content_leakage(tmp_path):
    raw = fixture_backup(tmp_path)
    index_path, split_path, pilot_path = [tmp_path / name for name in ("index.csv", "split.csv", "pilot.csv")]
    rows = build_index(raw, index_path)
    assert len(rows) == 4
    split = {row["asset_id"]: row for row in make_split(index_path, split_path)}
    # a and b share an event; a and c are identical content. The whole
    # connected component belongs to the latest partition, test.
    assert split["a"]["split"] == "exclude"
    assert split["b"]["split"] == "exclude"
    assert split["c"]["split"] == "test"
    assert split["d"]["split"] == "test"
    pilot = choose_pilot(index_path, split_path, pilot_path, count=10)
    assert {row["asset_id"] for row in pilot} == {"c", "d"}
    assert len({row["sha256"] for row in pilot}) == len(pilot)


def test_annotation_validation_and_export(tmp_path):
    raw = fixture_backup(tmp_path)
    index_path, split_path, ann_path = [tmp_path / name for name in ("index.csv", "split.csv", "annotations.jsonl")]
    index = build_index(raw, index_path)
    make_split(index_path, split_path)
    by_id = {row["asset_id"]: row for row in index}
    readable = {
        "asset_id": "c", "readability": "readable", "review_status": "reviewed",
        "annotator": "person-a", "reviewer": "person-b", "reason": "",
        "boxes": [{"xyxy": [15, 10, 80, 42], "view_kind": "scene", "text": "13.04"}],
    }
    negative = {
        "asset_id": "d", "readability": "no_display", "review_status": "reviewed",
        "annotator": "person-a", "reviewer": "person-b", "reason": "no scale",
        "boxes": [],
    }
    assert validate_annotation(readable, by_id["c"])["boxes"][0]["text"] == "13.04"
    test_row = {**by_id["c"], "split": "test"}
    with pytest.raises(ValueError, match="different reviewer"):
        validate_annotation({**readable, "reviewer": ""}, test_row)
    with pytest.raises(ValueError, match="different reviewer"):
        validate_annotation({**readable, "reviewer": "person-a"}, test_row)
    with pytest.raises(ValueError, match="annotator"):
        validate_annotation({**readable, "annotator": ""}, test_row)
    bad = {**readable, "boxes": [{"xyxy": [15, 10, 80, 42], "view_kind": "scene", "text": "1304?"}]}
    with pytest.raises(ValueError):
        validate_annotation(bad, by_id["c"])
    mismatched_zoom = {**readable, "boxes": readable["boxes"] + [
        {"xyxy": [20, 15, 90, 50], "view_kind": "zoom", "text": "13.40"}
    ]}
    with pytest.raises(ValueError, match="different numbers"):
        validate_annotation(mismatched_zoom, by_id["c"])
    ann_path.write_text("\n".join(json.dumps(a) for a in (readable, negative)) + "\n", encoding="utf-8")
    output = tmp_path / "prepared"
    card = export_dataset(raw, index_path, split_path, ann_path, output)
    assert card["counts"]["detector_images_test"] == 2
    assert card["counts"]["recognizer_crops_test"] == 1
    assert (output / "detector/labels/test/d.txt").read_text() == ""
    assert (output / "recognizer/test.txt").read_text().endswith("\t13.04\n")
    crop_map = read_csv(output / "crop_map.csv")
    assert crop_map[0]["group_id"]
    archive = tmp_path / "dataset.zip"
    digest = package_dataset(output, archive)
    assert digest in archive.with_name("dataset.zip.sha256").read_text()
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert "dataset_card.json" in bundle.namelist()
    with pytest.raises(FileExistsError):
        export_dataset(raw, index_path, split_path, ann_path, output)


def test_evaluation_counts_abstention_and_false_accept():
    rows = [
        {"sample_id": "1", "gateway": "g1", "readability": "readable", "status": "ok", "pred_text": "7.04", "gt_text": "7.04"},
        {"sample_id": "2", "gateway": "g1", "readability": "readable", "status": "review", "pred_text": "", "gt_text": "1.00"},
        {"sample_id": "3", "gateway": "g2", "readability": "unreadable", "status": "ok", "pred_text": "0.00", "gt_text": ""},
    ]
    report = summarize(rows)
    assert report["readable_exact_match"] == 0.5
    assert report["accepted_accuracy"] == 0.5
    assert report["false_accept_rate_unreadable"] == 1.0
    assert report["by_gateway"]["g1"]["counts"]["total"] == 2
