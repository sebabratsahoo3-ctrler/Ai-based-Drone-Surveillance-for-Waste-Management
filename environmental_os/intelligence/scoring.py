from __future__ import annotations

from dataclasses import dataclass

from environmental_os.schemas import EnvironmentalEvent, Severity


@dataclass
class CleanlinessScore:
    zone: str
    score: float
    grade: str

    @classmethod
    def compute(cls, zone: str, events: list[EnvironmentalEvent]) -> "CleanlinessScore":
        penalty = 0.0
        for event in events:
            if event.severity == Severity.MAJOR:
                penalty += 18
            elif event.severity == Severity.MODERATE:
                penalty += 8
            else:
                penalty += 3
        score = max(0.0, 100.0 - penalty)
        grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
        return cls(zone=zone, score=round(score, 1), grade=grade)
