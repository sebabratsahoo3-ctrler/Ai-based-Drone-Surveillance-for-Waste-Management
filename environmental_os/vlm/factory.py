from __future__ import annotations

import os
from pathlib import Path

from environmental_os.vlm.models import MockVLMReasoner, VLMReasoner


def create_vlm_reasoner() -> VLMReasoner:
    """Select VLM backend: fine-tuned HF LoRA > HTTP > mock."""
    finetuned = os.getenv("FINETUNED_VLM_PATH", "models/dronewaste-smolvlm-lora").strip()
    if finetuned and Path(finetuned).exists() and (Path(finetuned) / "adapter_config.json").exists():
        from environmental_os.vlm.hf_reasoner import HFVLMReasoner

        return HFVLMReasoner(adapter_path=finetuned)

    http_endpoint = os.getenv("VLM_HTTP_ENDPOINT", "").strip()
    if http_endpoint:
        from environmental_os.vlm.http_backend import HttpVLMReasoner

        return HttpVLMReasoner(
            endpoint_url=http_endpoint,
            model_name=os.getenv("VLM_HTTP_MODEL_NAME", "remote-vlm"),
        )

    return MockVLMReasoner()
