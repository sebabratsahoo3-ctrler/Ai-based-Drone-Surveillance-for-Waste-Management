CREATE TABLE IF NOT EXISTS drone_telemetry (
    telemetry_id TEXT PRIMARY KEY,
    drone_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    altitude_m REAL NOT NULL,
    speed_mps REAL,
    heading_deg REAL,
    battery_pct REAL,
    mission_id TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS environmental_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    event_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity TEXT NOT NULL,
    snapshot_ref TEXT NOT NULL,
    drone_altitude_m REAL NOT NULL,
    drone_id TEXT NOT NULL,
    scene_description TEXT NOT NULL,
    historical_relevance_score REAL NOT NULL,
    environmental_risk_level TEXT NOT NULL,
    escalation_status TEXT NOT NULL,
    rationale TEXT,
    involved_agents_json TEXT,
    temporal_evidence_json TEXT,
    correlated_event_ids_json TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS historical_intelligence (
    record_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    gps_zone TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS risk_analysis_logs (
    log_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    risk_score REAL NOT NULL,
    policy_version TEXT NOT NULL,
    reasoning_json TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES environmental_events(event_id)
);

CREATE TABLE IF NOT EXISTS incident_reports (
    report_id TEXT PRIMARY KEY,
    generated_at TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    recommendations_json TEXT NOT NULL,
    dispatch_targets_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS environmental_heatmaps (
    heatmap_id TEXT PRIMARY KEY,
    generated_at TEXT NOT NULL,
    gps_zone TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    event_count INTEGER NOT NULL,
    max_risk TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vector_memory (
    vector_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    gps_zone TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (record_id) REFERENCES historical_intelligence(record_id)
);

CREATE INDEX IF NOT EXISTS idx_events_time ON environmental_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_geo ON environmental_events(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_events_type ON environmental_events(event_type);
CREATE INDEX IF NOT EXISTS idx_history_zone ON historical_intelligence(gps_zone);
CREATE INDEX IF NOT EXISTS idx_vector_zone ON vector_memory(gps_zone);

