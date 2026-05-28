# RAG and Database Design

## RAG Retrieval Classes

The memory system retrieves:

- previous dumping incidents
- historical sanitation violations
- garbage hotspot history
- prior fire/smoke incidents
- crowd cleanliness patterns
- drain blockage history
- environmental hazard records
- repeated suspicious events
- risk evolution patterns

## Vector Memory

The starter implementation uses a small hash embedding store so it can run
without external dependencies. Production deployments should replace it with
SQLite vector extensions, FAISS, Qdrant, Milvus, pgvector, or another managed
vector database.

## Historical Context In Prompt

The VLM receives concise retrieved facts, for example:

```text
2026-05-05 night_dumping Previous illegal dumping reported near roadside drain.
2026-05-12 overflowing_bin Repeated waste accumulation near market entrance.
```

This helps the model reason beyond the current frame and explain why an event is
suspicious or severe.

## Tables

The schema includes:

- environmental_events
- historical_intelligence
- drone_telemetry
- risk_analysis_logs
- incident_reports
- environmental_heatmaps
- vector_memory

See `database/schema.sql`.

