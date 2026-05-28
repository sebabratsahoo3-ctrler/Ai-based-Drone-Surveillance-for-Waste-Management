from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    domain_randomization: list[str]
    expected_events: list[str]


SCENARIOS = [
    ScenarioSpec(
        name="market_overflowing_bins",
        description="Crowded market road with overflowing collection bins and roadside litter.",
        domain_randomization=["lighting", "crowd_density", "camera_angle", "altitude", "weather"],
        expected_events=["overflowing_bin", "roadside_littering", "sensitive_zone_accumulation"],
    ),
    ScenarioSpec(
        name="night_vehicle_dumping",
        description="Vehicle stops near a dark roadside drain and unloads waste bags.",
        domain_randomization=["night_lighting", "vehicle_density", "altitude", "camera_angle"],
        expected_events=["vehicle_dumping", "night_dumping", "drain_or_river_dumping"],
    ),
    ScenarioSpec(
        name="industrial_chemical_spill",
        description="Industrial waste area with visible liquid spill and nearby workers.",
        domain_randomization=["weather", "season", "camera_angle", "worker_density"],
        expected_events=["chemical_spill", "toxic_leakage", "hazardous_waste_exposure"],
    ),
    ScenarioSpec(
        name="waste_fire_smoke",
        description="Open waste burning with smoke plume near municipal dumping ground.",
        domain_randomization=["wind", "lighting", "smoke_density", "altitude"],
        expected_events=["smoke_or_fire", "waste_burning", "crowd_near_hazard"],
    ),
    ScenarioSpec(
        name="drain_blockage_waterlogging",
        description="Plastic waste blocks a roadside drain causing stagnant water.",
        domain_randomization=["rain", "water_level", "camera_angle", "crowd_density"],
        expected_events=["drain_blockage", "waterlogging", "environmental_contamination"],
    ),
]

