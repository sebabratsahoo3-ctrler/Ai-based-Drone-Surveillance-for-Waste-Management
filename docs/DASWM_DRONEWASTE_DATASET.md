# DASWM DroneWaste Dataset Integration

This project integrates the **DroneWaste v1.0** aerial waste dataset used in DASWM-related work.

## Source location (local)

Default path (Windows):

```text
D:\Most Related works to DASWM Project\Dataset DASWM
```

Contents:

| File / folder | Description |
|---------------|-------------|
| `images/` | 4993 aerial PNG frames (`site{N}_{id}.png`) |
| `dronewaste_v1.0.json` | COCO-style annotations (images, categories, bboxes, segmentation) |
| `info.txt` | Dataset metadata and category list |

Zenodo: https://doi.org/10.5281/zenodo.17045559

## Stats (from `info.txt`)

- **4993** images
- **5135** annotations
- **20** waste categories (EWC-coded)
- **17** sites

## How this OS uses it (VLM-centric)

The DroneWaste labels are **not** fed into YOLO-style detectors. Instead:

1. Category names per image are aggregated from COCO annotations.
2. Categories are mapped to Environmental OS `event_type` values (e.g. Asbestos → `hazardous_waste_exposure`).
3. Natural-language `scene_description` and VLM instruction/answer pairs are generated for fine-tuning and mock reasoning.

## Override dataset path

```powershell
$env:DRONEWASTE_DATASET_ROOT = "D:\Most Related works to DASWM Project\Dataset DASWM"
```

Or edit `configs/dronewaste.json`.

## Commands

Validate files (first 500 images):

```powershell
python scripts/build_dronewaste_vlm_dataset.py --validate
```

Build full VLM training JSON (annotated images only):

```powershell
python scripts/build_dronewaste_vlm_dataset.py --output data/synthetic/dronewaste_vlm_dataset.json
```

Quick test (100 samples):

```powershell
python scripts/build_dronewaste_vlm_dataset.py --limit 100
```

Run OS reasoning on real DroneWaste frames:

```powershell
python scripts/run_dronewaste_patrol_demo.py --limit 12
```

Output: `data/runtime/dronewaste_demo_output.json`

## Category → event mapping (summary)

| DroneWaste category | Environmental OS event |
|---------------------|-------------------------|
| Construction / Rubble / Asphalt | `construction_debris_dumping` |
| Vehicles | `vehicle_dumping` |
| Asbestos / E-waste | `hazardous_waste_exposure` |
| Plastic / Paper / Textile | `roadside_littering` |
| Mixed items | `large_illegal_dumping` |
| Foundry | `environmental_contamination` |

See `environmental_os/dataset/dronewaste.py` for the full mapping table.
