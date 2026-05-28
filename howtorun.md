# How to Run Environmental VLM OS (Practical)

This repo is a modular, VLM-centric OS for autonomous aerial waste management.
After fine-tuning, it loads **SmolVLM + LoRA** on DroneWaste automatically.

## 0. One-command project setup (recommended)

```powershell
cd "c:\Users\LENOVO\Documents\Ai based drone for waste management"
pip install -r requirement.txt
python scripts/prepare_project.py
```

This will:
1. Validate the DASWM DroneWaste dataset
2. Build `data/synthetic/dronewaste_vlm_dataset.json`
3. LoRA fine-tune SmolVLM (quick mode on CPU; use GPU for full training)
4. Run tests and a patrol demo

## 1. Prerequisites

- Python `>= 3.10`
- For the API + dashboard:
  - `pip install -r requirement.txt`

## 2. Install dependencies

```powershell
cd "c:\Users\LENOVO\Documents\Ai based drone for waste management"
pip install -r requirement.txt
```

## 3. Run the demo pipeline (edge + central reasoning)

This executes a multi-drone patrol loop, performs sparse scheduling + multimodal fusion,
writes `data/runtime/demo_output.json`, and prints the final JSON to the console.

```powershell
python -m environmental_os.demo
```

## 4. Run the central API + dashboard

Start the server:

```powershell
uvicorn environmental_os.central.api:app --reload --port 8000
```

Open the dashboard:
- http://localhost:8000/

Useful endpoints:
- `GET /health`
- `GET /events`
- `GET /heatmap`
- `GET /report`
- `GET /forecast`
- `GET /digital-twin`
- `POST /simulate/frame` (see payload fields below)
- `GET /dataset/dronewaste/info`
- `POST /dataset/dronewaste/process` (body: `{"file_name": "site10_10.png"}`)

### Simulate a frame payload

Example (Powershell):

```powershell
$payload = @{
  scene_hint = "smoke and fire near municipal waste accumulation area with crowd nearby"
  lighting = "night"
  thermal_hotspot_score = 0.9
  audio_anomaly_score = 0.7
  audio_labels = @("shouting")
  crowd_density = 0.75
  weather_risk = 0.25
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/simulate/frame" -ContentType "application/json" -Body $payload
```

## 5. DASWM DroneWaste dataset (real aerial images)

If you have the dataset at:

```text
D:\Most Related works to DASWM Project\Dataset DASWM
```

Set the path (optional if using the default above):

```powershell
$env:DRONEWASTE_DATASET_ROOT = "D:\Most Related works to DASWM Project\Dataset DASWM"
```

Validate and build VLM instruction JSON from `dronewaste_v1.0.json` + `images/`:

```powershell
python scripts/build_dronewaste_vlm_dataset.py --validate
python scripts/build_dronewaste_vlm_dataset.py --output data/synthetic/dronewaste_vlm_dataset.json
```

Run the OS on real DroneWaste frames (uses scene hints derived from annotations):

```powershell
python scripts/run_dronewaste_patrol_demo.py --limit 12
```

See [docs/DASWM_DRONEWASTE_DATASET.md](docs/DASWM_DRONEWASTE_DATASET.md) for category mapping and dataset details.

## 6. Fine-tune SmolVLM on DroneWaste (LoRA)

Build the full instruction dataset (all annotated images, ~1171 samples):

```powershell
python scripts/build_dronewaste_vlm_dataset.py --output data/synthetic/dronewaste_vlm_dataset.json
```

Quick fine-tune (80 samples, 40 steps — good for CPU smoke test):

```powershell
python scripts/finetune_dronewaste.py --quick
```

Full fine-tune (GPU recommended):

```powershell
python scripts/finetune_dronewaste.py
```

Output: `models/dronewaste-smolvlm-lora/`

Use the fine-tuned model:

```powershell
$env:FINETUNED_VLM_PATH = "models/dronewaste-smolvlm-lora"
python -m environmental_os.demo
```

## 7. Generate synthetic AirSim dataset (optional)

```powershell
python scripts/generate_synthetic_dataset.py
```

Outputs:
- `data/synthetic/airsim_synthetic_dataset.json`
- `data/synthetic/vlm_finetune_specs.json` (next step)

Export fine-tuning specs:

```powershell
python scripts/export_finetune_specs.py
```

## 8. Run tests

```powershell
python -m unittest discover tests
```

## 9. Using a remote VLM HTTP backend (optional)

The OS can call a remote VLM service via HTTP using `HttpVLMReasoner`.

Set environment variables before starting `uvicorn`:

```powershell
$env:VLM_HTTP_ENDPOINT = "http://YOUR_HOST:YOUR_PORT/your-vlm-endpoint"
$env:VLM_HTTP_MODEL_NAME = "phi3-vision-int4"   # optional
```

The remote endpoint must return JSON compatible with the structure expected by
`environmental_os.vlm.models.parse_vlm_json_response()`.

