from __future__ import annotations

import json
import re
import time
from pathlib import Path

from environmental_os.schemas import FrameContext, SceneObservation
from environmental_os.vlm.models import parse_vlm_json_response
from environmental_os.vlm.prompts import RAG_CONTEXT_PROMPT, SCENE_UNDERSTANDING_PROMPT, TEMPORAL_REASONING_PROMPT


class HFVLMReasoner:
    """Run fine-tuned (or base) SmolVLM with LoRA adapter for aerial waste reasoning."""

    def __init__(
        self,
        adapter_path: str | Path,
        base_model: str | None = None,
        max_new_tokens: int = 384,
    ):
        self.adapter_path = Path(adapter_path)
        self.max_new_tokens = max_new_tokens
        manifest_path = self.adapter_path / "training_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.base_model = base_model or manifest.get("base_model", "HuggingFaceTB/SmolVLM-256M-Instruct")
        else:
            self.base_model = base_model or "HuggingFaceTB/SmolVLM-256M-Instruct"
        self.model_name = f"hf-lora:{self.adapter_path.name}"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._Image = Image
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.adapter_path)
        base = AutoModelForImageTextToText.from_pretrained(self.base_model, torch_dtype=dtype)
        if (self.adapter_path / "adapter_config.json").exists():
            self._model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            self._model = base
        self._model.to(device)
        self._model.eval()
        self._device = device

    def analyze(
        self,
        frame: FrameContext,
        retrieved_context: list[str],
        temporal_context: list[str],
    ) -> SceneObservation:
        started = time.perf_counter()
        self._load()

        image_path = Path(frame.frame_ref)
        if image_path.exists() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            image = self._Image.open(image_path).convert("RGB")
        else:
            # No image file: return empty observation
            return SceneObservation(
                frame=frame,
                summary="No image available for VLM inference.",
                candidates=[],
                vlm_model=self.model_name,
                raw_reasoning={"error": "missing_image", "frame_ref": frame.frame_ref},
            )

        context_block = "\n".join(
            [
                SCENE_UNDERSTANDING_PROMPT.strip(),
                TEMPORAL_REASONING_PROMPT.strip(),
                RAG_CONTEXT_PROMPT.strip(),
                f"GPS zone: {frame.telemetry.gps.rounded_zone()}",
                f"Altitude: {frame.telemetry.altitude_m}m",
                f"Lighting: {frame.lighting}",
                f"Weather: {frame.weather}",
                f"Scene hint: {frame.metadata.get('scene_hint', '')}",
                "Retrieved history:\n" + "\n".join(retrieved_context[:4]),
                "Temporal memory:\n" + "\n".join(temporal_context[:4]),
            ]
        )
        prompt = f"User: {context_block}\nAssistant:"

        import torch

        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        json_payload = self._extract_json(text)

        try:
            observation = parse_vlm_json_response(json_payload, frame, self.model_name)
        except (json.JSONDecodeError, KeyError, ValueError):
            observation = SceneObservation(
                frame=frame,
                summary=text[:500],
                candidates=[],
                vlm_model=self.model_name,
                raw_reasoning={"raw_text": text, "parse_error": True},
            )

        observation.processing_ms = (time.perf_counter() - started) * 1000
        observation.raw_reasoning["hf_output_excerpt"] = text[-800:]
        return observation

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return '{"scene_summary": "", "event_candidates": []}'
