"""Build a reproducible index from the Cloudinary backup manifest."""

import csv
import hashlib
import json
from pathlib import Path


FIELDS = (
    "asset_id", "public_id", "local_file", "gateway", "path_date",
    "capture_kind", "event_uuid", "event_key", "slot", "schema_valid",
    "width", "height", "bytes", "uploaded_at", "sha256",
)


def parse_public_id(public_id: str) -> dict[str, str]:
    parts = public_id.split("/")
    if len(parts) not in (7, 9) or parts[0] != "roll-captures":
        return {"schema_valid": "false"}
    kind = parts[5]
    if not ((len(parts) == 7 and kind in {"core-weight", "product-weight"})
            or (len(parts) == 9 and kind == "photo-draft")):
        return {"schema_valid": "false"}
    try:
        from datetime import date
        day = date(int(parts[2]), int(parts[3]), int(parts[4])).isoformat()
    except ValueError:
        return {"schema_valid": "false"}
    if not parts[1] or not parts[6] or (len(parts) == 9 and not parts[8]):
        return {"schema_valid": "false"}
    return {
        "schema_valid": "true", "gateway": parts[1], "path_date": day,
        "capture_kind": kind, "event_uuid": parts[6],
        "event_key": f"{parts[1]}:{parts[6]}",
        "slot": parts[7] if len(parts) == 9 else "",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(raw_root: Path, destination: Path) -> list[dict[str, str]]:
    manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for asset in manifest["assets"]:
        public_id = asset["public_id"]
        if not public_id.startswith("roll-captures/"):
            continue
        relative = Path(asset["local_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe local_file: {relative}")
        source = raw_root / relative
        if not source.is_file() or source.stat().st_size != int(asset["bytes"]):
            raise ValueError(f"Missing or wrong size: {source}")
        row = {
            "asset_id": asset["asset_id"], "public_id": public_id,
            "local_file": relative.as_posix(),
            "width": str(asset["width"]), "height": str(asset["height"]),
            "bytes": str(asset["bytes"]),
            "uploaded_at": asset.get("uploaded_at", ""),
            "sha256": sha256_file(source),
        }
        row.update(parse_public_id(public_id))
        rows.append(row)
    rows.sort(key=lambda row: row["public_id"])
    if len({row["asset_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate asset_id in manifest")
    write_csv(destination, rows, FIELDS)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
