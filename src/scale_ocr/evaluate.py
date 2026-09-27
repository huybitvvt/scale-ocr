"""Evaluate whole-string readings without dropping abstentions."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from .inventory import read_csv


def summarize(rows: list[dict[str, str]]) -> dict:
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample_id must be unique")
    counts = Counter()
    by_gateway = defaultdict(list)
    for row in rows:
        by_gateway[row.get("gateway", "unknown")].append(row)
        readable = row["readability"] == "readable"
        accepted = row["status"] == "ok"
        correct = readable and accepted and row["pred_text"] == row["gt_text"]
        counts["total"] += 1
        counts["readable"] += int(readable)
        counts["accepted"] += int(accepted)
        counts["correct_accepted"] += int(correct)
        counts["unreadable"] += int(not readable)
        counts["false_accept_unreadable"] += int(not readable and accepted)

    def ratio(num, den):
        return num / den if den else None

    result = {
        "counts": dict(counts),
        "readable_exact_match": ratio(counts["correct_accepted"], counts["readable"]),
        "accepted_accuracy": ratio(counts["correct_accepted"], counts["accepted"]),
        "coverage_all": ratio(counts["accepted"], counts["total"]),
        "false_accept_rate_unreadable": ratio(
            counts["false_accept_unreadable"], counts["unreadable"]
        ),
    }
    if len(by_gateway) > 1:
        result["by_gateway"] = {key: summarize(group) for key, group in sorted(by_gateway.items())}
    return result


def evaluate_csv(predictions_path: Path, destination: Path) -> dict:
    result = summarize(read_csv(predictions_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
