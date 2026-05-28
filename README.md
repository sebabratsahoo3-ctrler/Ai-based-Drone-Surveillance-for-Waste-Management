# Centralized AI Aerial Waste Management OS

A modular, VLM-centric autonomous aerial environmental intelligence operating system for smart-city sanitation monitoring, environmental risk detection, and emergency escalation.

## Core Principles

- **No YOLO dependency**: reasoning is driven by lightweight Vision-Language Models.
- **Temporal intelligence first**: incidents are interpreted over multi-frame context, not isolated snapshots.
- **Edge-first deployment**: optimized for 4GB RAM edge hardware using sparse scheduling and quantized inference.
- **Central orchestration**: drone swarm events are fused, prioritized, correlated, and escalated via a centralized intelligence layer.

## Implemented Layers

1. **Drone Layer**
   - Autonomous patrol unit with waypoint routing
   - GPS-tagged frame buffering
   - Hazard-driven route adaptation hooks

2. **Edge AI Layer**
   - Sparse frame scheduling and dynamic resolution control
   - ROI policy (drain, bin, vehicle, smoke regions)
   - Async inference pipeline with edge caching
   - Multi-modal fusion hooks (vision + audio + thermal + weather + history)

3. **Central Intelligence Layer**
   - Cross-drone deduplication and swarm coordination
   - Severity/risk/escalation policy engine
   - Environmental heatmap generation
   - Digital twin zone state updates
   - Waste hotspot forecasting

4. **RAG Intelligence Layer**
   - Local lightweight vector memory for historical incident retrieval
   - Same-zone history boosting for repeated violations
   - Correlation with prior dumping/fire/sanitation incidents

5. **Persistence & Dashboard Layer**
   - SQLite persistence for events, reports, heatmaps, and historical records
   - FastAPI service endpoints for simulation, events, heatmaps, forecast, and digital twin
   - Dashboard prototype for incident and hotspot visualization

## Supported Lightweight VLM Targets

- Phi-3 Vision
- SmolVLM
- MiniCPM-V
- MobileVLM
- NanoVLM

The repository ships with `MockVLMReasoner` for deterministic local testing.

## Quick Start

Run demo pipeline:

```powershell
python -m environmental_os.demo
```

Generate synthetic training dataset:

```powershell
python scripts/generate_synthetic_dataset.py
```

Export fine-tuning specs:

```powershell
python scripts/export_finetune_specs.py
```

Run unit tests:

```powershell
python -m unittest discover tests
```

## API (FastAPI)

```powershell
uvicorn environmental_os.central.api:app --reload
```

Key endpoints:
- `POST /simulate/frame`
- `GET /events`
- `GET /heatmap`
- `GET /report`
- `GET /forecast`
- `GET /digital-twin`
- `GET /cleanliness`

## Data Model Coverage

Each event stores:
- timestamp
- GPS coordinates
- event type
- confidence
- severity classification
- snapshot reference
- drone altitude
- drone ID
- scene description
- historical relevance score
- environmental risk level
- escalation status

See `database/schema.sql` for complete schema.

## Edge Optimization Coverage

- INT4/INT8 model targets
- ONNX Runtime / TensorRT pathways
- sparse/event-triggered inference
- dynamic resolution scaling
- ROI-first processing
- async pipeline design
- lightweight vector embeddings
- edge cache

## DASWM DroneWaste Dataset (real aerial images)

Integrates the local DroneWaste v1.0 dataset (4993 images, COCO annotations) from:

`D:\Most Related works to DASWM Project\Dataset DASWM`

- Build VLM training JSON: `python scripts/build_dronewaste_vlm_dataset.py`
- Patrol demo on real frames: `python scripts/run_dronewaste_patrol_demo.py --limit 12`

See [docs/DASWM_DRONEWASTE_DATASET.md](docs/DASWM_DRONEWASTE_DATASET.md).

## Synthetic AirSim Scenario Coverage

Scenarios include market overflow, night vehicle dumping, smoke/fire, drain blockage waterlogging, and industrial chemical spill with domain randomization for lighting, weather, altitude, camera angle, and crowd/vehicle density.
