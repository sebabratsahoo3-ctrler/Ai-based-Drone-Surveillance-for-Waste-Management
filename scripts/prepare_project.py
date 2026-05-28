from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Environmental VLM OS for deployment.")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--quick-finetune", action="store_true", default=True)
    parser.add_argument("--dataset-limit", type=int, default=None, help="Limit VLM JSON samples (default: all annotated)")
    args = parser.parse_args()

    py = sys.executable

    run([py, "scripts/build_dronewaste_vlm_dataset.py", "--validate"], "Validate DroneWaste images")
    ds_cmd = [py, "scripts/build_dronewaste_vlm_dataset.py", "--output", "data/synthetic/dronewaste_vlm_dataset.json"]
    if args.dataset_limit:
        ds_cmd.extend(["--limit", str(args.dataset_limit)])
    run(ds_cmd, "Build DroneWaste VLM instruction dataset")

    if not args.skip_finetune:
        ft_cmd = [py, "scripts/finetune_dronewaste.py", "--config", "configs/finetune_dronewaste.json"]
        if args.quick_finetune:
            ft_cmd.append("--quick")
        run(ft_cmd, "LoRA fine-tune SmolVLM")

    run([py, "-m", "unittest", "discover", "tests"], "Run unit tests")
    run([py, "scripts/run_dronewaste_patrol_demo.py", "--limit", "4"], "Patrol demo on DroneWaste frames")

    print("\n=== Project ready ===")
    print("Start API: uvicorn environmental_os.central.api:app --reload --port 8000")
    print("Dashboard: http://localhost:8000/")


if __name__ == "__main__":
    main()
