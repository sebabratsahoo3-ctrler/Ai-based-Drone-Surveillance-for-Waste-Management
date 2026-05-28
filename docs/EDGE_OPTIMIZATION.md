# Edge Optimization

## Target Hardware

- Jetson Nano 4GB
- Raspberry Pi with AI accelerator
- Intel NCS2
- Coral TPU-supported devices

## Runtime Strategy

- INT4 or INT8 quantized VLMs
- ONNX Runtime, TensorRT, OpenVINO, or TFLite deployment
- sparse frame reasoning
- event-triggered inference
- dynamic resolution scaling
- ROI analysis
- compressed frame cache
- lightweight hash/vector embeddings
- asynchronous capture and inference
- low-power scheduling

## Recommended Model Use

| Model | Best Use |
| --- | --- |
| SmolVLM INT8 | default patrol on 4GB devices |
| MobileVLM INT8 | low-power edge patrol |
| NanoVLM INT8 | smallest fallback profile |
| MiniCPM-V INT4 | central edge verification |
| Phi-3 Vision INT4 | ROI verification or central node |

## Multi-Stage Reasoning

1. Capture low-resolution patrol frame.
2. Apply cheap motion, GPS, audio, thermal, or scenario cues.
3. Run VLM only on scheduled frames or suspicious windows.
4. Crop regions of interest for drains, bins, vehicles, smoke, or crowds.
5. Generate structured event candidates.
6. Stabilize across temporal memory.
7. Retrieve RAG history for the GPS zone.
8. Promote to event, risk, and escalation status.

