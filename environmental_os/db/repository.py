from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from environmental_os.rag.vector_store import HistoricalRecord
from environmental_os.schemas import EnvironmentalEvent, IncidentReport


class EnvironmentalDatabase:
    """SQLite persistence for events, telemetry, heatmaps, reports, and RAG memory."""

    def __init__(self, db_path: str | Path = "data/runtime/environmental_os.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        schema_path = Path(__file__).resolve().parents[2] / "database" / "schema.sql"
        sql = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(sql)

    def save_event(self, event: EnvironmentalEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO environmental_events (
                    event_id, timestamp, latitude, longitude, event_type, confidence,
                    severity, snapshot_ref, drone_altitude_m, drone_id, scene_description,
                    historical_relevance_score, environmental_risk_level, escalation_status,
                    rationale, involved_agents_json, temporal_evidence_json,
                    correlated_event_ids_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp,
                    event.gps.latitude,
                    event.gps.longitude,
                    event.event_type.value,
                    event.confidence,
                    event.severity.value,
                    event.snapshot_ref,
                    event.drone_altitude_m,
                    event.drone_id,
                    event.scene_description,
                    event.historical_relevance_score,
                    event.environmental_risk_level.value,
                    event.escalation_status.value,
                    event.rationale,
                    json.dumps(event.involved_agents),
                    json.dumps(event.temporal_evidence),
                    json.dumps(event.correlated_event_ids),
                    json.dumps(event.metadata),
                ),
            )

    def save_events(self, events: list[EnvironmentalEvent]) -> None:
        for event in events:
            self.save_event(event)

    def list_events(self, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM environmental_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_historical(self, record: HistoricalRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO historical_intelligence
                (record_id, timestamp, gps_zone, event_type, severity, text, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.timestamp,
                    record.gps_zone,
                    record.event_type,
                    record.severity,
                    record.text,
                    json.dumps(record.metadata),
                ),
            )

    def save_report(self, report: IncidentReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incident_reports
                (report_id, generated_at, title, summary, recommendations_json, dispatch_targets_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.generated_at,
                    report.title,
                    report.summary,
                    json.dumps(report.recommendations),
                    json.dumps(report.dispatch_targets),
                ),
            )

    def save_heatmap(self, heatmap: list[dict]) -> str:
        heatmap_id = f"hm-{uuid.uuid4().hex[:10]}"
        from datetime import datetime, timezone

        generated_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for item in heatmap:
                conn.execute(
                    """
                    INSERT INTO environmental_heatmaps
                    (heatmap_id, generated_at, gps_zone, latitude, longitude, event_count, max_risk, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        heatmap_id,
                        generated_at,
                        item["zone"],
                        item["latitude"],
                        item["longitude"],
                        item["count"],
                        item["max_risk"],
                        json.dumps(item),
                    ),
                )
        return heatmap_id
