import json
import tempfile
import unittest
from pathlib import Path

from environmental_os.dataset.synthetic import generate_synthetic_samples, write_samples
from environmental_os.demo import build_demo_orchestrator
from environmental_os.drone.patrol import DronePatrolUnit, PatrolWaypoint


class ExtendedSystemTests(unittest.TestCase):
    def test_synthetic_dataset_generation(self):
        config = json.loads(Path("configs/airsim_scenarios.json").read_text(encoding="utf-8"))
        samples = generate_synthetic_samples(config, count_per_scenario=3, seed=1)
        self.assertTrue(samples)
        self.assertEqual(len(samples), len(config["scenarios"]) * 3)
        self.assertIn("event_candidates", samples[0].answer)

        with tempfile.TemporaryDirectory() as tmp:
            path = write_samples(samples, Path(tmp) / "synthetic.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), len(samples))

    def test_drone_patrol_produces_frames(self):
        drone = DronePatrolUnit("drone-test")
        frames = list(
            drone.patrol_route(
                [
                    PatrolWaypoint(12.971, 77.594, scene_hint="overflowing bin near market"),
                    PatrolWaypoint(12.972, 77.595, scene_hint="smoke near waste zone", lighting="night"),
                ]
            )
        )
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[1].lighting, "night")

    def test_orchestrator_outputs_forecast_and_twin(self):
        orchestrator = build_demo_orchestrator()
        drone = DronePatrolUnit("drone-check")
        frame = drone.capture("chemical spill near industrial zone", altitude_m=36)
        events = orchestrator.process_frame(frame)
        self.assertTrue(events)
        self.assertTrue(orchestrator.forecast())
        self.assertTrue(orchestrator.digital_twin_snapshot())


if __name__ == "__main__":
    unittest.main()
