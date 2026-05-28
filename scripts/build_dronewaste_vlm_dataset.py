from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environmental_os.dataset.dronewaste import DronewasteDataset, DronewastePaths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert DroneWaste v1.0 to VLM instruction-tuning JSON.")
    parser.add_argument("--root", default=None, help="Dataset root (or set DRONEWASTE_DATASET_ROOT)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples (default: all annotated)")
    parser.add_argument("--include-unannotated", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Check image files exist")
    parser.add_argument(
        "--output",
        default="data/synthetic/dronewaste_vlm_dataset.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    paths = DronewastePaths.resolve(root=args.root)
    dataset = DronewasteDataset(paths)

    if args.validate:
        report = dataset.validate_files(limit=500)
        print(json.dumps(report, indent=2))
        if report["missing_count"] > 0:
            print("Warning: some images are missing on disk.")

    samples = dataset.to_vlm_samples(
        limit=args.limit,
        only_annotated=not args.include_unannotated,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    summary = {
        "output": str(out),
        "samples": len(samples),
        "dataset_root": str(paths.root),
        "categories": dataset.category_summary()[:5],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
