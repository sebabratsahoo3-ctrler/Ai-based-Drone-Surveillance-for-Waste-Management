# AirSim Simulation Plan

## Environment Objects

- roads
- public markets
- waste bins
- illegal dumping zones
- rivers and drainage systems
- industrial waste areas
- smoke and fire emitters
- waterlogging zones
- crowded sanitation zones
- hazardous waste areas
- nighttime dumping areas
- dynamic pedestrians
- vehicle-based dumping actors

## Scenario Families

1. Littering and plastic discard near roadside drains
2. Garbage thrown outside designated bins
3. Construction debris dumping
4. Overflowing bins near market entrances
5. Public spitting and hygiene violations
6. Waste burning with smoke plume
7. Chemical spill in industrial area
8. Sewage overflow and drain blockage
9. Vehicle dumping at night
10. Crowd gathering around hazardous waste

## Domain Randomization

- weather variation
- lighting variation
- drone altitude changes
- camera angle changes
- crowd density variation
- vehicle density variation
- seasonal conditions
- rain and water level
- smoke density
- market activity intensity

## Synthetic Dataset Output

Each generated sample should store:

- frame path
- timestamp
- drone pose
- GPS coordinate
- altitude
- lighting and weather
- scenario name
- VLM instruction
- structured expected answer
- temporal sequence ID
- severity label
- risk label

The labels are instruction-answer pairs for VLM fine-tuning, not bounding boxes.

