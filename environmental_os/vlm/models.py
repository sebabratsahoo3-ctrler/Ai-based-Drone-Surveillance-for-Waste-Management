from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

from environmental_os.schemas import EventCandidate, EventType, FrameContext, SceneObservation


@dataclass(frozen=True)
class ModelProfile:
    name: str
    family: str
    quantization: str
    backend: str
    max_image_px: int
    target_ram_mb: int
    notes: str


SUPPORTED_MODEL_PROFILES = {
    "phi3-vision-int4": ModelProfile(
        name="phi3-vision-int4",
        family="Phi-3 Vision",
        quantization="INT4",
        backend="ONNX Runtime / TensorRT",
        max_image_px=384 * 384,
        target_ram_mb=3900,
        notes="Use aggressive frame sparsity and ROI crops on 4GB devices.",
    ),
    "smolvlm-int8": ModelProfile(
        name="smolvlm-int8",
        family="SmolVLM",
        quantization="INT8",
        backend="ONNX Runtime",
        max_image_px=512 * 512,
        target_ram_mb=2200,
        notes="Default edge profile for continuous patrol analysis.",
    ),
    "minicpm-v-int4": ModelProfile(
        name="minicpm-v-int4",
        family="MiniCPM-V",
        quantization="INT4",
        backend="TensorRT",
        max_image_px=448 * 448,
        target_ram_mb=3600,
        notes="Best suited for central edge node verification.",
    ),
    "mobilevlm-int8": ModelProfile(
        name="mobilevlm-int8",
        family="MobileVLM",
        quantization="INT8",
        backend="ONNX Runtime",
        max_image_px=384 * 384,
        target_ram_mb=1800,
        notes="Low-power patrol mode.",
    ),
    "nanovlm-int8": ModelProfile(
        name="nanovlm-int8",
        family="NanoVLM",
        quantization="INT8",
        backend="ONNX Runtime",
        max_image_px=320 * 320,
        target_ram_mb=1200,
        notes="Fallback mode for tiny edge nodes.",
    ),
}


class VLMReasoner(Protocol):
    model_name: str

    def analyze(
        self,
        frame: FrameContext,
        retrieved_context: list[str],
        temporal_context: list[str],
    ) -> SceneObservation:
        ...


def parse_vlm_json_response(payload: str, frame: FrameContext, model_name: str) -> SceneObservation:
    data = json.loads(payload)
    candidates = []
    for item in data.get("event_candidates", []):
        event_type = EventType(item["event_type"])
        candidates.append(
            EventCandidate(
                event_type=event_type,
                confidence=float(item.get("confidence", 0.5)),
                scene_description=item.get("scene_description", ""),
                involved_agents=list(item.get("involved_agents", [])),
                rationale=item.get("rationale", ""),
                temporal_evidence=list(item.get("temporal_evidence", [])),
                snapshot_ref=frame.frame_ref,
                metadata={k: v for k, v in item.items() if k not in {
                    "event_type",
                    "confidence",
                    "scene_description",
                    "involved_agents",
                    "rationale",
                    "temporal_evidence",
                }},
            )
        )
    return SceneObservation(
        frame=frame,
        summary=data.get("scene_summary", ""),
        candidates=candidates,
        vlm_model=model_name,
        raw_reasoning=data,
    )


class MockVLMReasoner:
    """Deterministic VLM stand-in for local development and tests.

    Real deployments replace this with an adapter that calls Phi-3 Vision,
    SmolVLM, MiniCPM-V, MobileVLM, or NanoVLM through ONNX Runtime/TensorRT.
    """

    model_name = "mock-vlm-edge"

    def analyze(
        self,
        frame: FrameContext,
        retrieved_context: list[str],
        temporal_context: list[str],
    ) -> SceneObservation:
        started = time.perf_counter()
        visual_hint = " ".join(
            [
                str(frame.metadata.get("scene_hint", "")),
                " ".join(frame.roi_hints),
                frame.lighting,
                frame.weather,
            ]
        ).lower()
        history_hint = " ".join(retrieved_context).lower()
        event_prefixes = {event_type.value for event_type in EventType}
        zone_temporal_hint = " ".join(
            item
            for item in temporal_context
            if item.split(":", 1)[0] in event_prefixes
        ).lower()

        candidates: list[EventCandidate] = []

        def add(
            event_type: EventType,
            confidence: float,
            description: str,
            rationale: str,
            agents: list[str] | None = None,
        ) -> None:
            candidates.append(
                EventCandidate(
                    event_type=event_type,
                    confidence=confidence,
                    scene_description=description,
                    involved_agents=agents or [],
                    rationale=rationale,
                    temporal_evidence=temporal_context[-3:],
                    snapshot_ref=frame.frame_ref,
                )
            )

        if "toxic" in visual_hint or "chemical" in visual_hint or "spill" in visual_hint:
            add(
                EventType.CHEMICAL_SPILL if "spill" in visual_hint else EventType.TOXIC_LEAKAGE,
                0.86,
                "Possible hazardous liquid or chemical waste visible from aerial view.",
                "The scene contains spill or toxic leakage cues near a waste zone.",
                ["unknown handlers"] if "person" in visual_hint else [],
            )
        if "fire" in visual_hint or "smoke" in visual_hint or "burning" in visual_hint:
            add(
                EventType.SMOKE_OR_FIRE if "fire" in visual_hint or "smoke" in visual_hint else EventType.WASTE_BURNING,
                0.9,
                "Smoke or open burning detected near accumulated municipal waste.",
                "Smoke/fire cues indicate immediate environmental and public safety risk.",
            )
        if "overflow" in visual_hint or "overflowing bin" in visual_hint:
            add(
                EventType.OVERFLOWING_BIN,
                0.78,
                "Overflowing garbage bin detected with waste spread around the collection point.",
                "Waste outside the bin boundary suggests sanitation failure.",
            )
        if "drain" in visual_hint and ("garbage" in visual_hint or "block" in visual_hint or "water" in visual_hint):
            add(
                EventType.DRAIN_BLOCKAGE if "block" in visual_hint else EventType.DRAIN_OR_RIVER_DUMPING,
                0.76,
                "Waste material appears to obstruct or contaminate a drainage channel.",
                "Drain interaction creates waterlogging and contamination risk.",
            )
        if "waterlogging" in visual_hint or "stagnant water" in visual_hint or "flooded" in visual_hint:
            add(
                EventType.WATERLOGGING,
                0.79,
                "Water stagnation is visible around waste accumulation.",
                "Standing water near garbage suggests blocked drainage and disease risk.",
            )
        if "vehicle" in visual_hint and ("dump" in visual_hint or "throw" in visual_hint):
            large_dumping = any(
                token in visual_hint
                for token in ("large", "truck", "multiple bags", "construction debris", "bulk waste")
            )
            add(
                EventType.LARGE_ILLEGAL_DUMPING if large_dumping else EventType.VEHICLE_DUMPING,
                0.87 if large_dumping else 0.84,
                "Large illegal dumping activity observed from a vehicle."
                if large_dumping
                else "Suspicious dumping activity observed from a vehicle.",
                "Vehicle proximity, disposal motion, and waste accumulation indicate illegal dumping.",
                ["vehicle occupant"],
            )
        if "night" in visual_hint and ("dump" in visual_hint or "suspicious" in visual_hint):
            add(
                EventType.NIGHT_DUMPING,
                0.8,
                "Suspicious nighttime dumping behavior identified in a monitored zone.",
                "Low-light activity near waste hotspot increases suspiciousness.",
                ["unidentified individual"],
            )
        if "litter" in visual_hint or "plastic" in visual_hint or "discarded" in visual_hint:
            add(
                EventType.ROADSIDE_LITTERING,
                0.68,
                "An individual discarded plastic waste near the roadside.",
                "Small waste object appears newly placed outside a designated disposal point.",
                ["pedestrian"] if "person" in visual_hint or "individual" in visual_hint else [],
            )
        if "spitting" in visual_hint or "spit" in visual_hint:
            add(
                EventType.PUBLIC_SPITTING,
                0.64,
                "Public spitting behavior observed near a sanitation-sensitive zone.",
                "Human action is inconsistent with public hygiene rules.",
                ["pedestrian"],
            )
        if "crowd" in visual_hint and ("hazard" in visual_hint or "waste" in visual_hint or "panic" in visual_hint):
            add(
                EventType.PUBLIC_PANIC if "panic" in visual_hint else EventType.CROWD_NEAR_HAZARD,
                0.73,
                "Crowd activity is visible near a hazardous waste area.",
                "Crowd proximity raises exposure risk and may require area control.",
                ["crowd"],
            )
        if "school" in visual_hint or "hospital" in visual_hint or "market" in visual_hint:
            if "garbage" in visual_hint or "waste" in visual_hint:
                add(
                    EventType.SENSITIVE_ZONE_ACCUMULATION,
                    0.74,
                    "Garbage accumulation detected near a sensitive public zone.",
                    "Waste near schools, hospitals, or markets increases public health risk.",
                )
        same_zone_history = "same_zone" in history_hint and any(
            token in history_hint for token in ("repeated", "previous", "history", "hotspot")
        )
        same_zone_temporal = any(
            token in zone_temporal_hint
            for token in ("dump", "litter", "overflowing_bin", "drain_blockage", "waste")
        )
        visual_history_cue = any(token in visual_hint for token in ("repeated", "previous", "history", "hotspot"))
        if same_zone_history or same_zone_temporal or visual_history_cue:
            add(
                EventType.REPEATED_SANITATION_VIOLATION,
                0.82,
                "Repeated sanitation violations detected in this GPS zone.",
                "Historical context indicates this location has recurring incidents.",
            )

        if not candidates and ("garbage" in visual_hint or "waste" in visual_hint):
            add(
                EventType.TEMPORARY_WASTE_ACCUMULATION,
                0.55,
                "Temporary waste accumulation visible from drone view.",
                "Waste is present, but the scene lacks strong behavioral or hazard evidence.",
            )

        summary = candidates[0].scene_description if candidates else "No sanitation or environmental hazard event detected."
        return SceneObservation(
            frame=frame,
            summary=summary,
            candidates=candidates,
            vlm_model=self.model_name,
            processing_ms=(time.perf_counter() - started) * 1000,
            raw_reasoning={
                "mode": "deterministic_mock",
                "retrieved_context_count": len(retrieved_context),
                "temporal_context_count": len(temporal_context),
            },
        )


class LocalVLMAdapter:
    """Interface placeholder for quantized local VLM inference."""

    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self.model_name = profile.name

    def analyze(
        self,
        frame: FrameContext,
        retrieved_context: list[str],
        temporal_context: list[str],
    ) -> SceneObservation:
        raise NotImplementedError(
            "Wire this adapter to an ONNX Runtime, TensorRT, llama.cpp, or vendor "
            "runtime backend for the selected lightweight VLM."
        )
