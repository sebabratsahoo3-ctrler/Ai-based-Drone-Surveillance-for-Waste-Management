import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environmental_os.dataset.dronewaste import (
    CATEGORY_TO_EVENT,
    DronewasteDataset,
    DronewastePaths,
)


class DronewasteDatasetTests(unittest.TestCase):
    def test_default_path_config_loads(self):
        paths = DronewastePaths.resolve(
            config_path=Path("configs/dronewaste.json"),
        )
        self.assertTrue(paths.annotations.name == "dronewaste_v1.0.json")
        self.assertEqual(paths.images.name, "images")

    @patch("environmental_os.dataset.dronewaste.DEFAULT_DATASET_ROOT")
    def test_iter_records_with_tiny_mock_json(self, mock_root):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "site1_1.png").write_bytes(b"png")

            payload = {
                "info": {"stats": {"images": 1, "annotations": 1}},
                "categories": [
                    {"id": 1, "name": "Asbestos", "supercategory": "instance_waste", "ewc": "12.21"},
                ],
                "images": [
                    {"id": 1, "file_name": "site1_1.png", "site": "site1", "width": 640, "height": 640},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                ],
            }
            (root / "dronewaste_v1.0.json").write_text(json.dumps(payload), encoding="utf-8")

            paths = DronewastePaths(
                root=root,
                annotations=root / "dronewaste_v1.0.json",
                images=root / "images",
                info=root / "info.txt",
            )
            dataset = DronewasteDataset(paths)
            records = list(dataset.iter_records(limit=5))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].primary_event_type, CATEGORY_TO_EVENT["Asbestos"].value)
            self.assertIn("Hazardous", records[0].scene_description)

    def test_real_dataset_validate_if_present(self):
        paths = DronewastePaths.resolve()
        if not paths.annotations.exists():
            self.skipTest("DroneWaste dataset not available on this machine")
        dataset = DronewasteDataset(paths)
        report = dataset.validate_files(limit=20)
        self.assertGreater(report["checked"], 0)


if __name__ == "__main__":
    unittest.main()
