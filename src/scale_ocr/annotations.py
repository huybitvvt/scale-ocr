"""Local browser annotation tool for the pilot image set."""

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote

from .inventory import read_csv


STATUSES = {"readable", "partial", "unreadable", "no_display", "non_numeric"}
VIEWS = {"scene", "zoom", "full"}


def load_annotations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        asset_id = item["asset_id"]
        if asset_id in result:
            raise ValueError(f"Duplicate annotation for {asset_id} at line {line_number}")
        result[asset_id] = item
    return result


def save_annotations(path: Path, items: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for key in sorted(items):
            stream.write(json.dumps(items[key], ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_annotation(data: dict, row: dict) -> dict:
    if data.get("asset_id") != row["asset_id"]:
        raise ValueError("Asset ID mismatch")
    status = data.get("readability")
    if status not in STATUSES:
        raise ValueError("Invalid readability")
    boxes = data.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    if status == "readable" and not boxes:
        raise ValueError("A readable image needs at least one box")
    if status == "no_display" and boxes:
        raise ValueError("no_display cannot have boxes")
    width, height = int(row["width"]), int(row["height"])
    clean_boxes = []
    for box in boxes:
        coords = box.get("xyxy")
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError("Each box needs xyxy")
        x1, y1, x2, y2 = coords
        if any(isinstance(value, bool) or not isinstance(value, int) for value in coords):
            raise ValueError("Box coordinates must be integers")
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Box outside image")
        view = box.get("view_kind")
        if view not in VIEWS:
            raise ValueError("Invalid view kind")
        text = str(box.get("text", "")).strip()
        if status == "readable":
            import re
            if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
                raise ValueError("Text must be a visible numeric string; check decimal and minus")
        elif text:
            raise ValueError("Only readable boxes may have a numeric label")
        clean_boxes.append({"xyxy": coords, "view_kind": view, "text": text})
    if status == "readable" and len({box["text"] for box in clean_boxes}) > 1:
        raise ValueError("Scene and zoom show different numbers; review capture alignment")
    review_status = data.get("review_status", "draft")
    if review_status not in {"draft", "reviewed"}:
        raise ValueError("Invalid review status")
    annotator = str(data.get("annotator", "")).strip()[:100]
    reviewer = str(data.get("reviewer", "")).strip()[:100]
    if review_status == "reviewed" and not annotator:
        raise ValueError("Reviewed labels need an annotator")
    if review_status == "reviewed" and row.get("split") in {"val", "test"}:
        if not reviewer or reviewer == annotator:
            raise ValueError("Val/test labels need a different reviewer")
    return {
        "asset_id": row["asset_id"], "public_id": row["public_id"],
        "event_key": row["event_key"], "readability": status,
        "reason": str(data.get("reason", "")).strip()[:200],
        "annotator": annotator,
        "reviewer": reviewer,
        "review_status": review_status,
        "boxes": clean_boxes,
    }


def serve(pilot_path: Path, raw_root: Path, annotations_path: Path,
          host="127.0.0.1", port=8765) -> None:
    pilot = read_csv(pilot_path)
    by_asset = {row["asset_id"]: row for row in pilot}
    if len(by_asset) != len(pilot):
        raise ValueError("Duplicate asset in pilot")
    raw_root = raw_root.resolve()
    annotations = load_annotations(annotations_path)
    unknown = set(annotations) - set(by_asset)
    if unknown:
        raise ValueError(f"Annotations not present in pilot: {len(unknown)}")
    page = Path(__file__).with_name("annotate.html").read_bytes()
    save_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self.respond(200, page, "text/html; charset=utf-8")
            if self.path == "/api/items":
                return self.respond(200, [{
                    **row, "annotation": annotations.get(row["asset_id"])
                } for row in pilot])
            if self.path.startswith("/api/image/"):
                asset_id = unquote(self.path.removeprefix("/api/image/"))
                row = by_asset.get(asset_id)
                if row is None:
                    return self.respond(404, {"error": "Unknown asset"})
                target = (raw_root / row["local_file"]).resolve()
                if not target.is_relative_to(raw_root) or not target.is_file():
                    return self.respond(404, {"error": "Image missing"})
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                return self.respond(200, target.read_bytes(), content_type)
            self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/save":
                return self.respond(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100_000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                row = by_asset.get(data.get("asset_id"))
                if row is None:
                    raise ValueError("Asset not in pilot")
                clean = validate_annotation(data, row)
                with save_lock:
                    annotations[row["asset_id"]] = clean
                    save_annotations(annotations_path, annotations)
                self.respond(200, {"saved": row["asset_id"]})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})

    print(f"Open http://{host}:{port}/ ; labels: {annotations_path}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
