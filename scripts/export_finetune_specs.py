import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environmental_os.training.finetune import export_specs


def main() -> None:
    out = export_specs("data/synthetic/vlm_finetune_specs.json")
    print(f"Exported fine-tuning specs: {out}")


if __name__ == "__main__":
    main()
