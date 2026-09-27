import pytest

from tools.import_pilot_drafts import build_drafts


def test_audited_value_stays_draft_and_uses_manual_box():
    row = {
        "asset_id": "a", "public_id": "roll-captures/a", "event_key": "e",
        "split": "test", "width": "100", "height": "80",
    }
    candidate = {
        "index": 1, "asset_id": "a", "chosen": "",
        "scene_box": [10, 10, 40, 30], "scene_text": "8.42", "scene_conf": .8,
        "zoom_box": None, "zoom_text": "", "zoom_conf": 0,
    }
    audit = {"overrides": {"1": "0.42"}, "statuses": {}}
    manual = {"1": [{"xyxy": [12, 12, 42, 32], "view_kind": "scene"}]}
    draft = build_drafts([row], [candidate], audit, manual)["a"]
    assert draft["review_status"] == "draft"
    assert draft["boxes"] == [
        {"xyxy": [12, 12, 42, 32], "view_kind": "scene", "text": "0.42"}
    ]


def test_missing_value_does_not_become_ground_truth():
    row = {
        "asset_id": "a", "public_id": "roll-captures/a", "event_key": "e",
        "split": "train", "width": "100", "height": "80",
    }
    candidate = {
        "index": 1, "asset_id": "a", "chosen": "",
        "scene_box": None, "scene_text": "", "scene_conf": 0,
        "zoom_box": None, "zoom_text": "", "zoom_conf": 0,
    }
    with pytest.raises(ValueError, match="Missing visible number"):
        build_drafts([row], [candidate], {"overrides": {}, "statuses": {}}, {})
