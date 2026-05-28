# VLM Fine-Tuning Plan

## Fine-Tuning Targets

- aerial environmental understanding
- human sanitation behavior analysis
- environmental anomaly reasoning
- waste interaction understanding
- temporal scene intelligence
- small-object reasoning
- edge-optimized aerial surveillance

## Dataset Format

Each sample should be an image or short frame sequence with an instruction:

```json
{
  "image": "frames/market_overflow_001.jpg",
  "instruction": "Analyze this aerial sanitation scene and return structured environmental events.",
  "answer": {
    "scene_summary": "Overflowing garbage bin near market entrance.",
    "event_candidates": [
      {
        "event_type": "overflowing_bin",
        "confidence": 0.83,
        "scene_description": "Overflowing garbage bin detected near market entrance.",
        "rationale": "Waste has spread outside the collection point.",
        "temporal_evidence": ["similar waste spread visible in prior frames"]
      }
    ]
  }
}
```

## Optimization Techniques

- LoRA fine-tuning for task-specific reasoning
- quantization-aware training for INT8/INT4 deployment
- knowledge distillation from larger VLMs
- edge-oriented pruning
- short-context temporal memory optimization
- synthetic AirSim domain randomization

## Evaluation

Evaluate on:

- event type accuracy
- severity accuracy
- escalation accuracy
- GPS-zone historical reasoning
- temporal consistency
- false escalation rate
- edge latency
- RAM usage
- power draw

