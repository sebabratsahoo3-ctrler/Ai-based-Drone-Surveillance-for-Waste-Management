from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def load_vlm_samples(dataset_path: str | Path, max_samples: int | None = None) -> list[dict]:
    path = Path(dataset_path)
    samples = json.loads(path.read_text(encoding="utf-8"))
    valid = []
    for item in samples:
        image_path = Path(item["image_ref"])
        if not image_path.exists():
            continue
        valid.append(item)
        if max_samples and len(valid) >= max_samples:
            break
    return valid


def build_user_prompt(sample: dict, extra_context: str = "") -> str:
    instruction = sample.get("instruction", "Analyze this aerial environmental scene.")
    categories = sample.get("answer", {}).get("visible_waste_categories", [])
    site = sample.get("site", "unknown")
    parts = [
        instruction,
        f"Site: {site}.",
        f"Visible waste categories (metadata): {', '.join(categories) if categories else 'unknown'}.",
        "Return strict JSON only with keys: scene_summary, event_candidates.",
        "Each event_candidates item must include: event_type, confidence, scene_description, rationale, involved_agents, temporal_evidence.",
    ]
    if extra_context:
        parts.append(f"Context: {extra_context}")
    return "\n".join(parts)


def build_target_json(sample: dict) -> str:
    answer = sample.get("answer", {})
    payload = {
        "scene_summary": answer.get("scene_summary", ""),
        "event_candidates": answer.get("event_candidates", []),
    }
    return json.dumps(payload, ensure_ascii=False)


def load_training_pair(sample: dict) -> tuple[Image.Image, str, str]:
    image = Image.open(sample["image_ref"]).convert("RGB")
    user_text = build_user_prompt(sample)
    target_text = build_target_json(sample)
    return image, user_text, target_text
