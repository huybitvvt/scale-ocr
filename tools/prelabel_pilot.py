"""OCR-assisted draft proposals for the fixed-camera pilot.

This tool never marks an annotation as reviewed. Its JSON proposals and contact sheets
are candidates for visual checking, not numeric ground truth.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def numeric_text(text: str) -> str:
    """Normalize an OCR candidate for these displays' observed 2-decimal format."""
    text = text.strip().replace(",", ".")
    negative = text.startswith("-")
    digits = re.sub(r"\D", "", text)
    if not 2 <= len(digits) <= 5:
        return ""
    return ("-" if negative else "") + (digits[:-2] or "0") + "." + digits[-2:]


def region(image: np.ndarray, gateway: str, view: str) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    scene_end = min(height, 900)
    if view == "zoom":
        if height <= 950:
            return None
        return (round(width * .12), 930, round(width * .88), height)
    limits = {
        "gateway-01": (.34, .70, .62),
        "gateway-02": (.48, .97, .58),
        "gateway-03": (.32, .72, .60),
        "gateway-04": (.42, .85, .58),
    }
    left, right, top = limits.get(gateway, (.20, .95, .55))
    return (round(width * left), round(scene_end * top),
            round(width * right), scene_end)


def led_box(image: np.ndarray, gateway: str, view: str) -> list[int] | None:
    bounds = region(image, gateway, view)
    if bounds is None:
        return None
    x0, y0, x3, y3 = bounds
    sub = image[y0:y3, x0:x3]
    if sub.size == 0:
        return None
    blue, green, red = cv2.split(sub)
    red = red.astype(np.float32)
    green = green.astype(np.float32)
    blue = blue.astype(np.float32)
    mask = ((red > 85) & (red > green * 1.32) & (red > blue * 1.22)).astype("uint8")
    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (27, 15)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if width < 35 or height < 18 or not .55 < width / height < 8:
            continue
        original_red = int(mask[labels == index].sum())
        if original_red < 45:
            continue
        # Dense illuminated text outranks isolated lamps and red packaging.
        density = original_red / max(width * height, 1)
        score = original_red * min(width / 80, 2.0) * min(density / .10, 1.5)
        if view == "scene":
            score *= .5 + .5 * ((y0 + y + height / 2) / max(y3, 1))
        candidates.append((score, index, x, y, width, height))
    if not candidates:
        return None
    _, selected, x, y, width, height = max(candidates)
    # Use the selected connected component to trim the dilated halo, then pad.
    component = labels[y:y + height, x:x + width]
    yy, xx = np.where((component == selected) & (mask[y:y + height, x:x + width] > 0))
    if len(xx):
        x1, x2 = x + int(xx.min()), x + int(xx.max()) + 1
        y1, y2 = y + int(yy.min()), y + int(yy.max()) + 1
    else:
        x1, x2, y1, y2 = x, x + width, y, y + height
    padx = max(8, round((x2 - x1) * .08))
    pady = max(6, round((y2 - y1) * .12))
    return [max(x0, x0 + x1 - padx), max(y0, y0 + y1 - pady),
            min(x3, x0 + x2 + padx), min(y3, y0 + y2 + pady)]


def recognize(reader, image: np.ndarray, box: list[int] | None) -> tuple[str, float, str]:
    if box is None:
        return "", 0.0, ""
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0, ""
    variants = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), crop[:, :, 2]]
    scores = []
    for variant in variants:
        scaled = cv2.resize(variant, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        results = reader.recognize(scaled, allowlist="0123456789.-", detail=1)
        if results:
            _, raw, confidence = max(results, key=lambda item: item[2])
            value = numeric_text(raw)
            if value:
                scores.append((float(confidence), value, str(raw)))
    if not scores:
        return "", 0.0, ""
    confidence, value, raw = max(scores)
    return value, round(confidence, 4), raw


def font(size: int):
    candidate = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.exists() else ImageFont.load_default()


def paste_crop(canvas, image, box, x, y, label, width=190, height=94):
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), label, fill="black", font=font(17))
    y += 24
    if not box:
        draw.text((x, y + 20), "no LED box", fill="red", font=font(16))
        return
    x1, y1, x2, y2 = box
    crop = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    crop.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(crop, (x, y))


def contact_sheets(records, raw_root: Path, out_dir: Path, per_page=12):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, tile_w, tile_h = 3, 440, 158
    for start in range(0, len(records), per_page):
        page = records[start:start + per_page]
        rows = (len(page) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
        draw = ImageDraw.Draw(canvas)
        for offset, record in enumerate(page):
            x = offset % cols * tile_w + 8
            y = offset // cols * tile_h + 8
            image = cv2.imread(str(raw_root / record["local_file"]))
            title = f'{record["index"]:03d} {record["gateway"]} -> {record["chosen"] or "?"} {record["decision"]}'
            draw.text((x, y), title, fill="black", font=font(18))
            paste_crop(canvas, image, record["scene_box"], x, y + 22,
                       f'S {record["scene_text"]} ({record["scene_conf"]:.2f})')
            paste_crop(canvas, image, record["zoom_box"], x + 215, y + 22,
                       f'Z {record["zoom_text"]} ({record["zoom_conf"]:.2f})')
        canvas.save(out_dir / f"sheet_{start // per_page + 1:02d}.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    with args.pilot.open(encoding="utf-8", newline="") as stream:
        pilot = list(csv.DictReader(stream))
    records = []
    for index, row in enumerate(pilot, 1):
        image = cv2.imread(str(args.raw_root / row["local_file"]))
        if image is None:
            raise FileNotFoundError(row["local_file"])
        scene_box = led_box(image, row["gateway"], "scene")
        zoom_box = led_box(image, row["gateway"], "zoom")
        scene_text, scene_conf, scene_raw = recognize(reader, image, scene_box)
        zoom_text, zoom_conf, zoom_raw = recognize(reader, image, zoom_box)
        if scene_text and zoom_text and scene_text == zoom_text:
            chosen, decision = scene_text, "agree"
        elif scene_text and zoom_text:
            chosen, decision = "", "conflict"
        elif scene_text:
            chosen, decision = scene_text, "scene_only"
        elif zoom_text:
            chosen, decision = zoom_text, "zoom_only"
        else:
            chosen, decision = "", "no_read"
        records.append({
            "index": index, "asset_id": row["asset_id"], "local_file": row["local_file"],
            "public_id": row["public_id"], "gateway": row["gateway"], "split": row["split"],
            "scene_box": scene_box, "scene_text": scene_text, "scene_conf": scene_conf,
            "scene_raw": scene_raw, "zoom_box": zoom_box, "zoom_text": zoom_text,
            "zoom_conf": zoom_conf, "zoom_raw": zoom_raw, "chosen": chosen,
            "decision": decision,
        })
        if index % 25 == 0:
            print(f"Scanned {index}/{len(pilot)}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidates.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheets(records, args.raw_root, args.out_dir / "sheets")
    from collections import Counter
    print(dict(Counter(row["decision"] for row in records)), flush=True)


if __name__ == "__main__":
    main()
