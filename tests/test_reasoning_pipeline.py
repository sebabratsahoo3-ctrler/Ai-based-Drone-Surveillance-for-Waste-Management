import unittest

from environmental_os.airsim_bridge.client import AirSimDroneBridge
from environmental_os.central.orchestrator import EnvironmentalOrchestrator
from environmental_os.rag.vector_store import HistoricalRecord, LocalVectorStore
from environmental_os.schemas import EnvironmentalRisk, EscalationStatus, GPSPoint, Severity
from environmental_os.vlm.models import MockVLMReasoner


class ReasoningPipelineTests(unittest.TestCase):
    def test_major_smoke_event_auto_dispatches(self):
        orchestrator = EnvironmentalOrchestrator(vlm=MockVLMReasoner())
        bridge = AirSimDroneBridge("drone-test")
        frame = bridge.mock_frame(
            "smoke and fire near municipal waste accumulation area",
            gps=GPSPoint(12.1, 77.1),
        )

        events = orchestrator.process_frame(frame)

        self.assertTrue(events)
        event = events[0]
        self.assertEqual(event.severity, Severity.MAJOR)
        self.assertEqual(event.environmental_risk_level, EnvironmentalRisk.CRITICAL)
        self.assertEqual(event.escalation_status, EscalationStatus.AUTO_DISPATCH)

    def test_historical_context_increases_relevance(self):
        store = LocalVectorStore()
        store.add(
            HistoricalRecord(
                record_id="hist-1",
                timestamp="2026-05-01T00:00:00+00:00",
                gps_zone="12.9716:77.5946",
                text="Repeated illegal dumping near roadside drain during nighttime.",
                event_type="night_dumping",
                severity="Moderate Trigger",
                metadata={},
            )
        )
        orchestrator = EnvironmentalOrchestrator(vlm=MockVLMReasoner(), vector_store=store)
        bridge = AirSimDroneBridge("drone-test")
        frame = bridge.mock_frame(
            "vehicle stopped near roadside drain at night and person appears to dump garbage",
            gps=GPSPoint(12.9716, 77.5946),
            lighting="night",
        )

        events = orchestrator.process_frame(frame)

        self.assertTrue(any(event.historical_relevance_score > 0 for event in events))
        self.assertTrue(any(event.escalation_status == EscalationStatus.COMMAND_REVIEW for event in events))

    def test_repeated_zone_creates_repeated_violation(self):
        orchestrator = EnvironmentalOrchestrator(vlm=MockVLMReasoner())
        bridge = AirSimDroneBridge("drone-test")
        gps = GPSPoint(12.5, 77.5)

        for _ in range(3):
            events = orchestrator.process_frame(
                bridge.mock_frame("plastic garbage blocking drain with waste", gps=gps)
            )

        self.assertTrue(any(event.event_type.value == "repeated_sanitation_violation" for event in events))


if __name__ == "__main__":
    unittest.main()

