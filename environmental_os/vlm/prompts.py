SCENE_UNDERSTANDING_PROMPT = """
You are an aerial environmental intelligence VLM.

Analyze the drone frame without using object detector labels. Reason directly
from the visual scene, GPS context, altitude, lighting, weather, and recent
frame memory.

Return strict JSON with:
- scene_summary
- visible_people
- visible_vehicles
- sanitation_activity
- environmental_hazards
- why_suspicious
- danger_assessment
- event_candidates

Each event candidate must include:
- event_type
- confidence from 0 to 1
- involved_agents
- scene_description
- rationale
- temporal_evidence
"""


TEMPORAL_REASONING_PROMPT = """
Compare the current frame with the recent frame sequence. Decide whether this
is an isolated visual cue or a temporally supported event. Look for repeated
behavior, approach-dispose-leave patterns, worsening smoke, spreading water,
crowd movement, vehicle-based dumping, and recurring GPS-zone violations.
"""


RAG_CONTEXT_PROMPT = """
Use retrieved historical incident context to improve reasoning. If the same GPS
zone has repeated sanitation violations, similar nighttime activity, prior
fire/smoke events, drain blockage history, or garbage hotspot evolution, include
that context in the environmental risk explanation.
"""


EVENT_TYPES = [
    "garbage_dumping",
    "roadside_littering",
    "public_spitting",
    "outside_bin_waste",
    "drain_or_river_dumping",
    "construction_debris_dumping",
    "overflowing_bin",
    "public_urination",
    "waste_burning",
    "smoke_or_fire",
    "toxic_leakage",
    "sewage_overflow",
    "environmental_contamination",
    "drain_blockage",
    "waterlogging",
    "vehicle_dumping",
    "large_illegal_dumping",
    "night_dumping",
    "repeated_sanitation_violation",
    "crowd_near_hazard",
    "animals_scavenging",
    "hazardous_waste_exposure",
    "sensitive_zone_accumulation",
    "chemical_spill",
    "road_blockage",
    "waste_caused_accident",
    "fallen_electric_pole",
    "public_panic",
    "unusual_crowd_behavior",
    "temporary_waste_accumulation",
]
