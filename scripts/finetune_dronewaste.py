from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environmental_os.training.trainer import TrainConfig, train_dronewaste_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune SmolVLM on DroneWaste VLM dataset.")
    parser.add_argument("--config", default="configs/finetune_dronewaste.json")
    parser.add_argument("--quick", action="store_true", help="Small subset for smoke test / CPU")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config, quick=args.quick)
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.max_steps:
        config.max_steps = args.max_steps

    print(f"Training with {config.max_samples or 'all'} samples, device will be auto-detected...")
    manifest = train_dronewaste_lora(config)
    print(json.dumps(manifest, indent=2))
    print(f"\nSaved adapter to {config.output_dir}")
    print("Set FINETUNED_VLM_PATH=models/dronewaste-smolvlm-lora before running API/demo.")


if __name__ == "__main__":
    main()
