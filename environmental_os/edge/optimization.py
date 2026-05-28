from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationPlan:
    model_name: str
    quantization: str
    runtime: str
    max_resolution: tuple[int, int]
    frame_strategy: str
    memory_strategy: str
    notes: list[str]


EDGE_OPTIMIZATION_PLANS = {
    "jetson_nano_4gb": OptimizationPlan(
        model_name="smolvlm-int8",
        quantization="INT8",
        runtime="ONNX Runtime with TensorRT execution provider",
        max_resolution=(512, 512),
        frame_strategy="sparse patrol frames plus event-triggered bursts",
        memory_strategy="keep 8 to 12 frame summaries and compressed thumbnails",
        notes=[
            "Prefer SmolVLM or MobileVLM profile for continuous patrol.",
            "Run Phi-3 Vision only on ROI crops or central edge node verification.",
            "Use TensorRT engine cache and pinned memory.",
        ],
    ),
    "raspberry_pi_accelerator_4gb": OptimizationPlan(
        model_name="nanovlm-int8",
        quantization="INT8",
        runtime="ONNX Runtime or accelerator vendor runtime",
        max_resolution=(384, 384),
        frame_strategy="low-power interval inference with hazard keyword triggers",
        memory_strategy="summary-only temporal memory with lightweight hash embeddings",
        notes=[
            "Keep VLM context short.",
            "Offload central verification for major events.",
            "Use dynamic resolution scaling under thermal pressure.",
        ],
    ),
    "intel_ncs2": OptimizationPlan(
        model_name="mobilevlm-int8",
        quantization="INT8",
        runtime="OpenVINO / ONNX Runtime",
        max_resolution=(448, 448),
        frame_strategy="ROI crops and batched low-frequency patrol frames",
        memory_strategy="local vector cache with periodic central sync",
        notes=[
            "Convert visual encoder to OpenVINO IR where possible.",
            "Keep language decoder small and streamed.",
        ],
    ),
    "coral_tpu_supported": OptimizationPlan(
        model_name="nanovlm-int8",
        quantization="INT8",
        runtime="TFLite Edge TPU for auxiliary encoders plus CPU decoder",
        max_resolution=(320, 320),
        frame_strategy="tiny overview frames plus triggered high-priority crops",
        memory_strategy="edge cache of embeddings and event summaries",
        notes=[
            "Use TPU for lightweight pre-embedding or image encoding stages.",
            "Escalate uncertain reasoning to central intelligence node.",
        ],
    ),
}

