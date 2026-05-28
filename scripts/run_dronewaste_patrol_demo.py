from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from environmental_os.dataset.dronewaste import DronewasteDataset
from environmental_os.demo import build_demo_orchestrator
from environmental_os.schemas import DroneTelemetry, FrameContext, GPSPoint
from datetime import datetime, timezone


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run VLM OS reasoning on real DroneWaste images.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    dataset = DronewasteDataset()
    orchestrator = build_demo_orchestrator()

    all_events = []
    for record in dataset.iter_records(limit=args.limit, only_annotated=True):
        if not record.image_path.exists():
            continue
        lat, lon = map(float, record.gps_zone.split(":"))
        telemetry = DroneTelemetry(
            drone_id="drone-dronewaste",
            timestamp=datetime.now(timezone.utc).isoformat(),
            gps=GPSPoint(lat, lon),
            altitude_m=42.0,
            mission_id="dronewaste-patrol",
        )
        frame = FrameContext(
            frame_id=f"dronewaste-{record.image_id}",
            frame_ref=str(record.image_path),
            telemetry=telemetry,
            metadata={
                "scene_hint": record.scene_description,
                "dronewaste_categories": record.category_names,
                "site": record.site,
                "source": "dronewaste_v1.0",
            },
        )
        events = orchestrator.process_frame(frame)
        all_events.extend(events)

    output = {
        "dataset": "dronewaste_v1.0",
        "processed_frames": args.limit,
        "events": [e.to_dict() for e in all_events],
        "heatmap": orchestrator.heatmap(),
    }
    out_path = Path("data/runtime/dronewaste_demo_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
