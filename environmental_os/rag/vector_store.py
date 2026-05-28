from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class HistoricalRecord:
    record_id: str
    timestamp: str
    gps_zone: str
    text: str
    event_type: str
    severity: str
    metadata: dict


@dataclass
class SearchResult:
    record: HistoricalRecord
    score: float


class HashEmbedding:
    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = [token.strip(".,;:!?()[]{}").lower() for token in text.split()]
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        return [value / norm for value in vec]


class LocalVectorStore:
    """Tiny JSON-backed vector store for edge RAG and tests."""

    def __init__(self, path: str | Path | None = None, embedding: HashEmbedding | None = None):
        self.path = Path(path) if path else None
        self.embedding = embedding or HashEmbedding()
        self.records: list[HistoricalRecord] = []
        self.vectors: list[list[float]] = []
        if self.path and self.path.exists():
            self.load()

    def add(self, record: HistoricalRecord) -> None:
        self.records.append(record)
        self.vectors.append(self.embedding.embed(self._record_text(record)))

    def add_many(self, records: Iterable[HistoricalRecord]) -> None:
        for record in records:
            self.add(record)

    def search(self, query: str, gps_zone: str | None = None, limit: int = 5) -> list[SearchResult]:
        if not self.records:
            return []
        query_vec = self.embedding.embed(query)
        scored = []
        for record, vector in zip(self.records, self.vectors):
            score = self._cosine(query_vec, vector)
            if gps_zone and record.gps_zone == gps_zone:
                score += 0.18
            scored.append(SearchResult(record=record, score=min(score, 1.0)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in self.records]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.path:
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = [HistoricalRecord(**item) for item in payload]
        self.vectors = [self.embedding.embed(self._record_text(record)) for record in self.records]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(left * right for left, right in zip(a, b))

    @staticmethod
    def _record_text(record: HistoricalRecord) -> str:
        return f"{record.gps_zone} {record.event_type} {record.severity} {record.text}"

