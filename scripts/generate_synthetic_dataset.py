from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environmental_os.dataset.synthetic import generate_synthetic_samples, write_samples


def main() -> None:
    config_path = Path("configs/airsim_scenarios.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    samples = generate_synthetic_samples(config, count_per_scenario=24, seed=42)
    output_path = write_samples(samples, "data/synthetic/airsim_synthetic_dataset.json")
    print(f"Generated {len(samples)} synthetic samples at {output_path}")


if __name__ == "__main__":
    main()
