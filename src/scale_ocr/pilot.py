"""Select diverse, nonduplicate annotation candidates."""

import random
from collections import defaultdict
from pathlib import Path

from .inventory import read_csv, write_csv


FIELDS = ("asset_id", "public_id", "local_file", "gateway", "path_date",
          "capture_kind", "event_key", "group_id", "split", "sha256", "width", "height")


def choose_pilot(index_path: Path, split_path: Path, destination: Path,
                 count: int = 400, seed: int = 42) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("count must be positive")
    splits = {row["asset_id"]: row for row in read_csv(split_path)}
    buckets = defaultdict(list)
    seen_hashes = set()
    for row in read_csv(index_path):
        split = splits[row["asset_id"]]
        if split["split"] == "exclude" or row["sha256"] in seen_hashes:
            continue
        seen_hashes.add(row["sha256"])
        combined = {**row, "group_id": split["group_id"], "split": split["split"]}
        key = (combined["split"], combined["gateway"], combined["capture_kind"])
        buckets[key].append(combined)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    keys = sorted(buckets)
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            if len(selected) >= count:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    write_csv(destination, selected, FIELDS)
    return selected
