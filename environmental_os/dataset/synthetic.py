from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from environmental_os.schemas import EventType, Severity


@dataclass
class SyntheticSample:
    sample_id: str
    image_ref: str
    timestamp: str
    gps: dict
    altitude_m: float
    lighting: str
    weather: str
    scenario: str
    instruction: str
    answer: dict
    temporal_sequence_id: str
    severity: str
    risk: str

    def to_dict(self) -> dict:
        return asdict(self)


def generate_synthetic_samples(
    scenario_config: dict,
    count_per_scenario: int = 20,
    seed: int = 7,
) -> list[SyntheticSample]:
    rng = random.Random(seed)
    domain = list(scenario_config.get("domain_randomization", []))
    scenarios = list(scenario_config.get("scenarios", []))

    lighting_pool = ["day", "night", "dusk", "foggy_day"]
    weather_pool = ["clear", "rain", "cloudy", "haze"]
    base_time = datetime.now(timezone.utc) - timedelta(days=3)

    samples: list[SyntheticSample] = []
    for scenario in scenarios:
        scenario_name = scenario["name"]
        prompt = scenario["prompt"]
        events = scenario["events"]
        sequence_id = f"seq-{scenario_name}"

        for idx in range(count_per_scenario):
            lat = 12.9700 + rng.random() * 0.01
            lon = 77.5930 + rng.random() * 0.01
            altitude = round(25 + rng.random() * 40, 2)
            timestamp = (base_time + timedelta(minutes=len(samples) * 3)).isoformat()
            lighting = rng.choice(lighting_pool)
            weather = rng.choice(weather_pool)
            primary = events[idx % len(events)]

            severity = _severity_for(primary)
            risk = _risk_for(severity)
            rationale = (
                f"Scenario {scenario_name} with {primary} cues under {lighting}/{weather}. "
                f"Randomization factors: {', '.join(rng.sample(domain, k=min(3, len(domain))))}."
            )

            sample = SyntheticSample(
                sample_id=f"{scenario_name}-{idx:04d}",
                image_ref=f"airsim://synthetic/{scenario_name}/frame_{idx:04d}.png",
                timestamp=timestamp,
                gps={"latitude": round(lat, 6), "longitude": round(lon, 6)},
                altitude_m=altitude,
                lighting=lighting,
                weather=weather,
                scenario=scenario_name,
                instruction="Analyze this aerial environmental scene and output structured incident reasoning.",
                answer={
                    "scene_summary": prompt,
                    "event_candidates": [
                        {
                            "event_type": primary,
                            "confidence": round(0.62 + rng.random() * 0.33, 3),
                            "scene_description": prompt,
                            "rationale": rationale,
                            "temporal_evidence": [
                                f"{primary} seen in previous frame context",
                                "GPS zone has related sanitation history",
                            ],
                        }
                    ],
                },
                temporal_sequence_id=sequence_id,
                severity=severity,
                risk=risk,
            )
            samples.append(sample)

    return samples


def write_samples(samples: list[SyntheticSample], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.to_dict() for sample in samples]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _severity_for(event_name: str) -> str:
    event_type = EventType(event_name)
    if event_type in {
        EventType.TOXIC_LEAKAGE,
        EventType.SMOKE_OR_FIRE,
        EventType.HAZARDOUS_WASTE_EXPOSURE,
        EventType.LARGE_ILLEGAL_DUMPING,
        EventType.WASTE_BURNING,
        EventType.CHEMICAL_SPILL,
        EventType.SEWAGE_OVERFLOW,
    }:
        return Severity.MAJOR.value
    if event_type in {
        EventType.OVERFLOWING_BIN,
        EventType.REPEATED_SANITATION_VIOLATION,
        EventType.PUBLIC_SPITTING,
        EventType.GARBAGE_DUMPING,
        EventType.DRAIN_BLOCKAGE,
        EventType.NIGHT_DUMPING,
        EventType.VEHICLE_DUMPING,
    }:
        return Severity.MODERATE.value
    return Severity.MINOR.value


def _risk_for(severity: str) -> str:
    if severity == Severity.MAJOR.value:
        return "critical"
    if severity == Severity.MODERATE.value:
        return "high"
    return "low"
