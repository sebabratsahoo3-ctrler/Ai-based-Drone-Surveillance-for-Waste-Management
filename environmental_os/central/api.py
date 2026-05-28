from __future__ import annotations

import os
from pathlib import Path

from datetime import datetime, timezone

from environmental_os.airsim_bridge.client import AirSimDroneBridge
from environmental_os.central.orchestrator import EnvironmentalOrchestrator
from environmental_os.dataset.dronewaste import DronewasteDataset
from environmental_os.db.repository import EnvironmentalDatabase
from environmental_os.fusion.multimodal import SensorBundle
from environmental_os.geo.fence import GeoFence, GeoFenceMonitor
from environmental_os.intelligence.scoring import CleanlinessScore
from environmental_os.reporting import build_incident_report
from environmental_os.schemas import DroneTelemetry, FrameContext, GPSPoint
from environmental_os.vlm.factory import create_vlm_reasoner


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Install the api extra: pip install -e .[api]") from exc

    app = FastAPI(title="Environmental VLM OS", version="0.1.0")
    monitor = GeoFenceMonitor(
        fences=[
            GeoFence("market-core", "Market Zone", 12.9700, 12.9735, 77.5930, 77.5960, sensitive=True),
            GeoFence("hospital-buffer", "Hospital Buffer", 12.9730, 12.9755, 77.5950, 77.5975, sensitive=True),
        ]
    )
    orchestrator = EnvironmentalOrchestrator(
        vlm=create_vlm_reasoner(),
        geofence_monitor=monitor,
        database=EnvironmentalDatabase(),
    )
    bridge = AirSimDroneBridge()

    static_root = Path(__file__).resolve().parents[2] / "dashboard"
    app.mount("/dashboard", StaticFiles(directory=static_root), name="dashboard")

    @app.get("/", include_in_schema=False)
    def dashboard_root():
        # Serve the dashboard from the same FastAPI origin as the API.
        return FileResponse(static_root / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "vlm": orchestrator.vlm.model_name}

    @app.post("/simulate/frame")
    def simulate_frame(payload: dict):
        gps = GPSPoint(
            latitude=float(payload.get("latitude", 12.9716)),
            longitude=float(payload.get("longitude", 77.5946)),
        )
        frame = bridge.mock_frame(
            scene_hint=str(payload.get("scene_hint", "overflowing bin near market")),
            gps=gps,
            altitude_m=float(payload.get("altitude_m", 42)),
            lighting=str(payload.get("lighting", "day")),
        )
        sensors = SensorBundle(
            frame=frame,
            audio_anomaly_score=float(payload.get("audio_anomaly_score", 0.0)),
            audio_labels=list(payload.get("audio_labels", [])),
            thermal_hotspot_score=float(payload.get("thermal_hotspot_score", 0.0)),
            thermal_labels=list(payload.get("thermal_labels", [])),
            weather_risk=float(payload.get("weather_risk", 0.0)),
            crowd_density=float(payload.get("crowd_density", 0.0)),
        )
        events = orchestrator.process_frame(frame, sensors=sensors)
        return {"events": [event.to_dict() for event in events]}

    @app.get("/events")
    def events():
        return {"events": [event.to_dict() for event in orchestrator.events]}

    @app.get("/heatmap")
    def heatmap():
        return {"heatmap": orchestrator.heatmap()}

    @app.get("/report")
    def report():
        return build_incident_report(orchestrator.events).to_dict()

    @app.get("/forecast")
    def forecast():
        return {"forecast": orchestrator.forecast()}

    @app.get("/digital-twin")
    def digital_twin():
        return {"digital_twin": orchestrator.digital_twin_snapshot()}

    @app.get("/cleanliness")
    def cleanliness():
        zones: dict[str, list] = {}
        for event in orchestrator.events:
            zones.setdefault(event.gps.rounded_zone(), []).append(event)
        scores = [CleanlinessScore.compute(zone, zone_events).__dict__ for zone, zone_events in zones.items()]
        return {"scores": sorted(scores, key=lambda item: item["score"])}

    @app.get("/dataset/dronewaste/info")
    def dronewaste_info():
        try:
            dataset = DronewasteDataset()
            dataset.load()
            validation = dataset.validate_files(limit=100)
            return {
                "available": True,
                "root": str(dataset.paths.root),
                "stats": dataset.stats,
                "categories": dataset.category_summary()[:10],
                "validation": validation,
            }
        except FileNotFoundError as exc:
            return {"available": False, "error": str(exc)}

    @app.post("/dataset/dronewaste/process")
    def dronewaste_process(payload: dict):
        file_name = str(payload.get("file_name", "")).strip()
        if not file_name:
            return {"error": "file_name is required (e.g. site10_10.png)"}
        dataset = DronewasteDataset()
        record = None
        for item in dataset.iter_records(only_annotated=False):
            if item.file_name == file_name:
                record = item
                break
        if record is None:
            return {"error": f"image not found in annotations: {file_name}"}
        lat, lon = map(float, record.gps_zone.split(":"))
        frame = FrameContext(
            frame_id=f"dronewaste-{record.image_id}",
            frame_ref=str(record.image_path),
            telemetry=DroneTelemetry(
                drone_id="drone-dronewaste",
                timestamp=datetime.now(timezone.utc).isoformat(),
                gps=GPSPoint(lat, lon),
                altitude_m=float(payload.get("altitude_m", 42)),
            ),
            metadata={
                "scene_hint": record.scene_description,
                "dronewaste_categories": record.category_names,
                "site": record.site,
            },
        )
        events = orchestrator.process_frame(frame)
        return {
            "file_name": file_name,
            "scene_description": record.scene_description,
            "categories": record.category_names,
            "events": [event.to_dict() for event in events],
        }

    return app

try:
    app = create_app()
except RuntimeError:
    app = None
