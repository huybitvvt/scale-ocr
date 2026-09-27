"""Time split with event and exact-content leakage controls."""

from collections import Counter, defaultdict
from pathlib import Path

from .inventory import read_csv, write_csv


FIELDS = ("asset_id", "group_id", "split", "reason", "path_date", "gateway", "capture_kind", "sha256")
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def desired_split(day: str, train_end: str, val_end: str, test_end: str) -> str:
    if day <= train_end:
        return "train"
    if day <= val_end:
        return "val"
    if day <= test_end:
        return "test"
    return "exclude"


def make_split(index_path: Path, destination: Path, train_end="2026-09-24",
               val_end="2026-09-25", test_end="2026-09-27") -> list[dict[str, str]]:
    if not train_end < val_end < test_end:
        raise ValueError("Expected train_end < val_end < test_end")
    rows = read_csv(index_path)
    if len({row["asset_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate asset IDs")
    # The station-01 files in this snapshot are only four reused QR/black
    # placeholders. Their hashes are excluded across every gateway before
    # joining duplicate groups, or they could connect hundreds of events.
    placeholder_hashes = {r["sha256"] for r in rows if r["gateway"] == "station-01"}
    by_event = defaultdict(list)
    by_hash = defaultdict(list)
    dsu = DisjointSet()
    excluded = {}
    for row in rows:
        asset = row["asset_id"]
        if row["schema_valid"] != "true":
            excluded[asset] = "invalid_schema"
            continue
        if row["sha256"] in placeholder_hashes:
            excluded[asset] = "station_placeholder_content"
            continue
        if desired_split(row["path_date"], train_end, val_end, test_end) == "exclude":
            excluded[asset] = "outside_date_window"
            continue
        dsu.find(asset)
        by_event[row["event_key"]].append(asset)
        by_hash[row["sha256"]].append(asset)
    for groups in (by_event, by_hash):
        for assets in groups.values():
            for asset in assets[1:]:
                dsu.union(assets[0], asset)
    grouped = defaultdict(list)
    for row in rows:
        if row["asset_id"] not in excluded:
            grouped[dsu.find(row["asset_id"])].append(row)
    output = []
    for row in rows:
        asset = row["asset_id"]
        if asset in excluded:
            split, group_id, reason = "exclude", "", excluded[asset]
        else:
            members = grouped[dsu.find(asset)]
            group_id = min(member["asset_id"] for member in members)
            splits = [desired_split(member["path_date"], train_end, val_end, test_end)
                      for member in members]
            split = max(splits, key=SPLIT_ORDER.__getitem__)
            own_split = desired_split(row["path_date"], train_end, val_end, test_end)
            reason = "" if own_split == split else f"linked_to_later_{split}"
            if reason:
                split = "exclude"
        output.append({
            "asset_id": asset, "group_id": group_id, "split": split,
            "reason": reason, "path_date": row.get("path_date", ""),
            "gateway": row.get("gateway", ""),
            "capture_kind": row.get("capture_kind", ""),
            "sha256": row["sha256"],
        })
    write_csv(destination, output, FIELDS)
    active = [row for row in output if row["split"] in SPLIT_ORDER]
    by_group_split = defaultdict(set)
    by_hash_split = defaultdict(set)
    for row in active:
        by_group_split[row["group_id"]].add(row["split"])
        by_hash_split[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in by_group_split.values()):
        raise AssertionError("Group leakage")
    if any(len(splits) > 1 for splits in by_hash_split.values()):
        raise AssertionError("Exact image leakage")
    return output


def summarize_splits(rows):
    return dict(Counter(row["split"] for row in rows))
