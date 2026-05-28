from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from environmental_os.schemas import EventType, Severity


DEFAULT_DATASET_ROOT = Path(
    os.getenv(
        "DRONEWASTE_DATASET_ROOT",
        r"D:\Most Related works to DASWM Project\Dataset DASWM",
    )
)


# Map DroneWaste EWC categories to Environmental OS event types (VLM-centric, not bbox labels).
CATEGORY_TO_EVENT: dict[str, EventType] = {
    "Rubble": EventType.CONSTRUCTION_DEBRIS_DUMPING,
    "Construction and demolition materials": EventType.CONSTRUCTION_DEBRIS_DUMPING,
    "Asphalt milling": EventType.CONSTRUCTION_DEBRIS_DUMPING,
    "Excavation materials": EventType.CONSTRUCTION_DEBRIS_DUMPING,
    "Appliances": EventType.OUTSIDE_BIN_WASTE,
    "Electronic equipment": EventType.HAZARDOUS_WASTE_EXPOSURE,
    "Furniture": EventType.OUTSIDE_BIN_WASTE,
    "Metal barrels": EventType.TOXIC_LEAKAGE,
    "Plastic packaging": EventType.ROADSIDE_LITTERING,
    "Wood": EventType.TEMPORARY_WASTE_ACCUMULATION,
    "Pallets": EventType.TEMPORARY_WASTE_ACCUMULATION,
    "Scrap": EventType.GARBAGE_DUMPING,
    "Plastic": EventType.ROADSIDE_LITTERING,
    "Vehicles": EventType.VEHICLE_DUMPING,
    "Tyres": EventType.TEMPORARY_WASTE_ACCUMULATION,
    "Paper": EventType.ROADSIDE_LITTERING,
    "Foundry": EventType.ENVIRONMENTAL_CONTAMINATION,
    "Asbestos": EventType.HAZARDOUS_WASTE_EXPOSURE,
    "Textile": EventType.ROADSIDE_LITTERING,
    "Mixed items": EventType.LARGE_ILLEGAL_DUMPING,
}

CATEGORY_SEVERITY_HINT: dict[str, Severity] = {
    "Asbestos": Severity.MAJOR,
    "Electronic equipment": Severity.MAJOR,
    "Metal barrels": Severity.MAJOR,
    "Vehicles": Severity.MODERATE,
    "Construction and demolition materials": Severity.MODERATE,
    "Mixed items": Severity.MODERATE,
}


@dataclass(frozen=True)
class DronewastePaths:
    root: Path
    annotations: Path
    images: Path
    info: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None, config_path: str | Path | None = None) -> "DronewastePaths":
        if config_path:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            root = Path(cfg["dataset_root"])
        else:
            root = Path(root) if root else DEFAULT_DATASET_ROOT
        return cls(
            root=root,
            annotations=root / "dronewaste_v1.0.json",
            images=root / "images",
            info=root / "info.txt",
        )


@dataclass
class DronewasteImageRecord:
    image_id: int
    file_name: str
    site: str
    width: int
    height: int
    image_path: Path
    category_names: list[str]
    supercategories: list[str]
    primary_event_type: str
    scene_description: str
    severity: str
    gps_zone: str

    def to_vlm_sample(self) -> dict:
        return {
            "sample_id": f"dronewaste-{self.image_id}",
            "image_ref": str(self.image_path),
            "file_name": self.file_name,
            "site": self.site,
            "gps_zone": self.gps_zone,
            "instruction": (
                "Analyze this aerial drone waste monitoring image. "
                "Return structured environmental intelligence JSON with scene_summary and event_candidates."
            ),
            "answer": {
                "scene_summary": self.scene_description,
                "visible_waste_categories": self.category_names,
                "supercategories": list(set(self.supercategories)),
                "event_candidates": [
                    {
                        "event_type": self.primary_event_type,
                        "confidence": 0.82,
                        "scene_description": self.scene_description,
                        "rationale": (
                            f"Aerial view shows {', '.join(self.category_names[:4])} "
                            f"at {self.site} (DroneWaste v1.0)."
                        ),
                        "involved_agents": [],
                        "temporal_evidence": [],
                    }
                ],
            },
            "severity": self.severity,
            "dataset": "dronewaste_v1.0",
        }


class DronewasteDataset:
    """Loader for the DASWM DroneWaste v1.0 COCO-style aerial waste dataset."""

    def __init__(self, paths: DronewastePaths | None = None):
        self.paths = paths or DronewastePaths.resolve()
        self._data: dict | None = None
        self._categories_by_id: dict[int, dict] = {}
        self._images_by_id: dict[int, dict] = {}
        self._anns_by_image: dict[int, list[dict]] = defaultdict(list)

    def load(self) -> dict:
        if self._data is None:
            if not self.paths.annotations.exists():
                raise FileNotFoundError(f"Annotations not found: {self.paths.annotations}")
            self._data = json.loads(self.paths.annotations.read_text(encoding="utf-8"))
            self._categories_by_id = {c["id"]: c for c in self._data["categories"]}
            self._images_by_id = {img["id"]: img for img in self._data["images"]}
            for ann in self._data["annotations"]:
                self._anns_by_image[ann["image_id"]].append(ann)
        return self._data

    @property
    def stats(self) -> dict:
        data = self.load()
        return data.get("info", {}).get("stats", data.get("info", {}))

    def validate_files(self, limit: int | None = None) -> dict:
        self.load()
        missing = []
        checked = 0
        for img in self._data["images"]:
            if limit and checked >= limit:
                break
            path = self.paths.images / img["file_name"]
            if not path.exists():
                missing.append(img["file_name"])
            checked += 1
        return {
            "checked": checked,
            "missing_count": len(missing),
            "missing_sample": missing[:10],
            "images_dir": str(self.paths.images),
        }

    def _site_gps_zone(self, site: str) -> str:
        match = re.search(r"(\d+)", site)
        site_num = int(match.group(1)) if match else 1
        lat = 12.9700 + (site_num * 0.0015)
        lon = 77.5930 + (site_num * 0.0012)
        return f"{round(lat, 4)}:{round(lon, 4)}"

    def _build_scene_description(self, category_names: list[str], site: str, supercats: set[str]) -> str:
        if not category_names:
            return f"Aerial sanitation patrol over {site} with no labeled waste instances in metadata."
        primary = category_names[0]
        if "pile_waste" in supercats and len(category_names) >= 2:
            return (
                f"Large waste accumulation detected at {site}: "
                f"{', '.join(category_names[:3])} visible from drone view."
            )
        if "Vehicles" in category_names or "Tyres" in category_names:
            return f"Discarded vehicle-related waste observed at {site} from aerial view."
        if "Asbestos" in category_names or "Electronic equipment" in category_names:
            return f"Hazardous or regulated waste exposure visible at {site} from aerial view."
        return f"Aerial view at {site} shows {primary} and related waste materials."

    def _primary_event(self, category_names: list[str]) -> tuple[EventType, Severity]:
        if not category_names:
            return EventType.TEMPORARY_WASTE_ACCUMULATION, Severity.MINOR
        events = [CATEGORY_TO_EVENT.get(name, EventType.TEMPORARY_WASTE_ACCUMULATION) for name in category_names]
        severities = [CATEGORY_SEVERITY_HINT.get(name, Severity.MINOR) for name in category_names]
        # Prefer highest-severity mapping
        order = {Severity.MAJOR: 3, Severity.MODERATE: 2, Severity.MINOR: 1}
        best_name = max(category_names, key=lambda n: order.get(CATEGORY_SEVERITY_HINT.get(n, Severity.MINOR), 1))
        return CATEGORY_TO_EVENT.get(best_name, events[0]), max(severities, key=lambda s: order[s])

    def iter_records(
        self,
        only_annotated: bool = True,
        sites: list[str] | None = None,
        limit: int | None = None,
    ):
        self.load()
        count = 0
        for img in self._data["images"]:
            if sites and img["site"] not in sites:
                continue
            anns = self._anns_by_image.get(img["id"], [])
            if only_annotated and not anns:
                continue
            cat_ids = [a["category_id"] for a in anns]
            cat_counter = Counter(cat_ids)
            category_names = [
                self._categories_by_id[cid]["name"]
                for cid, _ in cat_counter.most_common()
            ]
            supercats = {
                self._categories_by_id[cid]["supercategory"]
                for cid in cat_ids
                if cid in self._categories_by_id
            }
            event_type, severity = self._primary_event(category_names)
            record = DronewasteImageRecord(
                image_id=img["id"],
                file_name=img["file_name"],
                site=img["site"],
                width=img["width"],
                height=img["height"],
                image_path=self.paths.images / img["file_name"],
                category_names=category_names,
                supercategories=list(supercats),
                primary_event_type=event_type.value,
                scene_description=self._build_scene_description(category_names, img["site"], supercats),
                severity=severity.value,
                gps_zone=self._site_gps_zone(img["site"]),
            )
            yield record
            count += 1
            if limit and count >= limit:
                break

    def to_vlm_samples(self, limit: int | None = None, only_annotated: bool = True) -> list[dict]:
        return [r.to_vlm_sample() for r in self.iter_records(limit=limit, only_annotated=only_annotated)]

    def category_summary(self) -> list[dict]:
        self.load()
        counts: Counter[str] = Counter()
        for ann in self._data["annotations"]:
            name = self._categories_by_id[ann["category_id"]]["name"]
            counts[name] += 1
        return [{"category": k, "annotations": v} for k, v in counts.most_common()]
