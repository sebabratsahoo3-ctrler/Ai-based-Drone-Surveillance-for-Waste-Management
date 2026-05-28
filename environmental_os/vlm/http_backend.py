from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from environmental_os.schemas import FrameContext, SceneObservation
from environmental_os.vlm.models import VLMReasoner, parse_vlm_json_response


@dataclass(frozen=True)
class HttpVLMConfig:
    endpoint_url: str
    model_name: str = "remote-vlm"
    timeout_s: float = 45.0
    auth_header: str | None = None
    auth_token: str | None = None


class HttpVLMReasoner:
    """Call a remote lightweight VLM service and parse VLM JSON output.

    The remote service is expected to return JSON in the same structure
    as `parse_vlm_json_response()` expects (scene_summary + event_candidates).
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str = "remote-vlm",
        timeout_s: float = 45.0,
        auth_header: str | None = None,
        auth_token: str | None = None,
    ):
        self.config = HttpVLMConfig(
            endpoint_url=endpoint_url,
            model_name=model_name,
            timeout_s=timeout_s,
            auth_header=auth_header,
            auth_token=auth_token,
        )
        self.model_name = model_name

    def analyze(
        self,
        frame: FrameContext,
        retrieved_context: list[str],
        temporal_context: list[str],
    ) -> SceneObservation:
        payload: dict[str, Any] = {
            "frame_ref": frame.frame_ref,
            "scene_hint": frame.metadata.get("scene_hint", ""),
            "roi_hints": frame.roi_hints,
            "lighting": frame.lighting,
            "weather": frame.weather,
            "drone_altitude_m": frame.telemetry.altitude_m,
            "gps": {
                "latitude": frame.telemetry.gps.latitude,
                "longitude": frame.telemetry.gps.longitude,
            },
            "retrieved_context": retrieved_context,
            "temporal_context": temporal_context,
            "max_candidates": 6,
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.auth_header and self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token

        req = urllib.request.Request(
            self.config.endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
            text = resp.read().decode("utf-8")

        # The parser converts VLM output into SceneObservation + EventCandidate objects.
        return parse_vlm_json_response(text, frame=frame, model_name=self.model_name)

