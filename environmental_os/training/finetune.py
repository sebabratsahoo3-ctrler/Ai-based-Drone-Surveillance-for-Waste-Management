from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FinetuneSpec:
    model_family: str
    method: str
    quantization_aware: bool
    distillation_teacher: str
    target_device: str
    context_frames: int
    lora_rank: int
    learning_rate: float
    epochs: int
    prune_ratio: float


DEFAULT_SPECS = [
    FinetuneSpec(
        model_family="Phi-3 Vision",
        method="LoRA",
        quantization_aware=True,
        distillation_teacher="larger_vlm_teacher",
        target_device="jetson_nano_4gb",
        context_frames=6,
        lora_rank=16,
        learning_rate=2e-4,
        epochs=3,
        prune_ratio=0.25,
    ),
    FinetuneSpec(
        model_family="SmolVLM",
        method="LoRA + distillation",
        quantization_aware=True,
        distillation_teacher="phi3_vision_finetuned",
        target_device="raspberry_pi_ai_4gb",
        context_frames=5,
        lora_rank=12,
        learning_rate=2.5e-4,
        epochs=4,
        prune_ratio=0.2,
    ),
    FinetuneSpec(
        model_family="MobileVLM",
        method="QAT + pruning",
        quantization_aware=True,
        distillation_teacher="smolvlm_teacher",
        target_device="intel_ncs2",
        context_frames=4,
        lora_rank=8,
        learning_rate=3e-4,
        epochs=4,
        prune_ratio=0.3,
    ),
]


def build_default_specs() -> list[FinetuneSpec]:
    return list(DEFAULT_SPECS)


def export_specs(path: str | Path) -> Path:
    payload = [asdict(spec) for spec in DEFAULT_SPECS]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
