"""Entry points for dataset preparation and review."""

import argparse
import json
from pathlib import Path

from .annotations import serve
from .evaluate import evaluate_csv
from .export import export_dataset
from .inventory import build_index
from .package import package_dataset
from .pilot import choose_pilot
from .splits import make_split, summarize_splits


def main(argv=None):
    parser = argparse.ArgumentParser(prog="scale-ocr")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Hash and index images in the backup")
    index.add_argument("--raw-root", type=Path, required=True)
    index.add_argument("--out", type=Path, required=True)
    split = sub.add_parser("split", help="Split by time, event and exact image")
    split.add_argument("--index", type=Path, required=True)
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--train-end", default="2026-09-24")
    split.add_argument("--val-end", default="2026-09-25")
    split.add_argument("--test-end", default="2026-09-27")
    pilot = sub.add_parser("pilot", help="Select balanced annotation candidates")
    pilot.add_argument("--index", type=Path, required=True)
    pilot.add_argument("--split", type=Path, required=True)
    pilot.add_argument("--out", type=Path, required=True)
    pilot.add_argument("--count", type=int, default=400)
    pilot.add_argument("--seed", type=int, default=42)
    annotate = sub.add_parser("annotate", help="Run local browser label tool")
    annotate.add_argument("--pilot", type=Path, required=True)
    annotate.add_argument("--raw-root", type=Path, required=True)
    annotate.add_argument("--out", type=Path, required=True)
    annotate.add_argument("--port", type=int, default=8765)
    export = sub.add_parser("export", help="Export reviewed labels for training")
    export.add_argument("--raw-root", type=Path, required=True)
    export.add_argument("--index", type=Path, required=True)
    export.add_argument("--split", type=Path, required=True)
    export.add_argument("--annotations", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--margin", type=float, default=0.05)
    package = sub.add_parser("package", help="Bundle exported dataset for Drive")
    package.add_argument("--source", type=Path, required=True)
    package.add_argument("--out", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate predictions CSV")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "index":
        rows = build_index(args.raw_root, args.out)
        print(json.dumps({"images": len(rows), "unique_sha256": len({r['sha256'] for r in rows})}))
    elif args.command == "split":
        rows = make_split(args.index, args.out, args.train_end, args.val_end, args.test_end)
        print(json.dumps(summarize_splits(rows)))
    elif args.command == "pilot":
        rows = choose_pilot(args.index, args.split, args.out, args.count, args.seed)
        print(json.dumps({"pilot_images": len(rows)}))
    elif args.command == "annotate":
        serve(args.pilot, args.raw_root, args.out, port=args.port)
    elif args.command == "export":
        print(json.dumps(export_dataset(
            args.raw_root, args.index, args.split, args.annotations, args.out, args.margin
        )["counts"], indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_csv(args.predictions, args.out), indent=2))
    elif args.command == "package":
        print(json.dumps({"archive": str(args.out), "sha256": package_dataset(args.source, args.out)}))


if __name__ == "__main__":
    main()
