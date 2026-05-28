from __future__ import annotations

import asyncio
import json
from pathlib import Path

from environmental_os.central.orchestrator import EnvironmentalOrchestrator
from environmental_os.db.repository import EnvironmentalDatabase
from environmental_os.drone.patrol import DronePatrolUnit, PatrolWaypoint
from environmental_os.edge.pipeline import EdgeInferencePipeline
from environmental_os.fusion.multimodal import SensorBundle
from environmental_os.geo.fence import GeoFence, GeoFenceMonitor
from environmental_os.rag.vector_store import HistoricalRecord, LocalVectorStore
from environmental_os.reporting import build_incident_report
from environmental_os.schemas import GPSPoint, IncidentReport
from environmental_os.vlm.factory import create_vlm_reasoner


def build_demo_orchestrator() -> EnvironmentalOrchestrator:
    store = LocalVectorStore()
    store.add_many(
        [
            HistoricalRecord(
                record_id="hist-001",
                timestamp="2026-05-05T20:10:00+00:00",
                gps_zone="12.9716:77.5946",
                text="Previous illegal dumping reported near roadside drain during nighttime.",
                event_type="night_dumping",
                severity="Moderate Trigger",
                metadata={"source": "municipal_log"},
            ),
            HistoricalRecord(
                record_id="hist-002",
                timestamp="2026-05-12T18:40:00+00:00",
                gps_zone="12.9716:77.5946",
                text="Garbage hotspot history shows repeated waste accumulation near market entrance.",
                event_type="overflowing_bin",
                severity="Moderate Trigger",
                metadata={"source": "dashboard"},
            ),
            HistoricalRecord(
                record_id="hist-003",
                timestamp="2026-05-19T07:05:00+00:00",
                gps_zone="12.9721:77.5951",
                text="Smoke detected near municipal waste accumulation area.",
                event_type="smoke_or_fire",
                severity="Major Trigger",
                metadata={"source": "drone_patrol"},
            ),
        ]
    )
    return EnvironmentalOrchestrator(
        vlm=create_vlm_reasoner(),
        vector_store=store,
        geofence_monitor=GeoFenceMonitor(
            fences=[
                GeoFence("market-core", "Market Zone", 12.9700, 12.9735, 77.5930, 77.5960, sensitive=True),
                GeoFence("river-drain", "Drain Buffer", 12.9725, 12.9740, 77.5950, 77.5970, sensitive=False),
            ]
        ),
        database=EnvironmentalDatabase(),
    )


async def run_demo() -> dict:
    orchestrator = build_demo_orchestrator()
    drone_alpha = DronePatrolUnit("drone-alpha")
    drone_beta = DronePatrolUnit("drone-beta")
    pipeline = EdgeInferencePipeline(
        vlm=orchestrator.vlm,
        process_fn=orchestrator.process_frame,
        sensor_resolver=lambda frame: SensorBundle(
            frame=frame,
            thermal_hotspot_score=0.88 if "fire" in frame.metadata["scene_hint"] else 0.32,
            audio_anomaly_score=0.75 if "crowd" in frame.metadata["scene_hint"] else 0.2,
            audio_labels=["shouting"] if "crowd" in frame.metadata["scene_hint"] else [],
            weather_risk=0.65 if "waterlogging" in frame.metadata["scene_hint"] else 0.25,
        ),
    )

    routes = [
        (
            drone_alpha,
            [
            PatrolWaypoint(12.9716, 77.5946, scene_hint="market entrance with overflowing bin, plastic litter"),
            PatrolWaypoint(
                12.9716,
                77.5946,
                lighting="night",
                scene_hint="vehicle stopped near roadside drain at night, person appears to dump garbage bags",
            ),
            PatrolWaypoint(12.9721, 77.5951, scene_hint="smoke and fire near municipal waste accumulation area"),
            ],
        ),
        (
            drone_beta,
            [
            PatrolWaypoint(12.9730, 77.5960, scene_hint="plastic garbage blocking drain, stagnant waterlogging visible"),
            PatrolWaypoint(
                12.9732,
                77.5962,
                scene_hint="possible toxic spill in industrial waste area with crowd nearby",
            ),
            ],
        ),
    ]

    all_events = []
    for drone, waypoints in routes:
        for frame in drone.patrol_route(waypoints):
            all_events.extend(await pipeline.submit(frame))

    report: IncidentReport = build_incident_report(all_events)
    heatmap = orchestrator.heatmap()
    return {
        "events": [event.to_dict() for event in all_events],
        "heatmap": heatmap,
        "forecast": orchestrator.forecast(),
        "digital_twin": orchestrator.digital_twin_snapshot(),
        "edge_stats": pipeline.stats.__dict__,
        "report": report.to_dict(),
    }


def main() -> None:
    output = asyncio.run(run_demo())
    output_path = Path("data/runtime/demo_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()

