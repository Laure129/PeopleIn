import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from peoplein.counter import DoorCounter


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = iter(boxes)

    def __call__(self, _frame):
        return [(box, 0.9) for box in next(self.boxes)]


class ScoredDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def __call__(self, _frame):
        return next(self.detections)


class DoorCounterTest(unittest.TestCase):
    def test_evidence_has_three_frames_before_and_after_for_each_camera(self):
        directions = {
            "left_to_right": "entry",
            "right_to_left": "exit",
        }
        cameras = {
            camera: {
                "line": ((0, 0), (0, 20)),
                "directions": directions,
            }
            for camera in ("entrance", "loby")
        }
        boxes = []
        for tick in range(7):
            boxes.extend((
                [(-12 if tick < 3 else 8, 0, 4, 10)],
                [],
            ))
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        started = datetime(2026, 1, 2)

        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            counter = DoorCounter(
                cameras=cameras,
                model_path="unused",
                confidence=0.35,
                agreement_seconds=15,
                crossing_margin_px=1,
                database_path=Path(directory) / "people.sqlite3",
                app_version="test",
                detector=FakeDetector(boxes),
                evidence_dir=evidence,
            )
            for tick in range(7):
                timestamp = started + timedelta(seconds=tick)
                for camera in cameras:
                    counter.update(camera, frame, timestamp)

            self.assertEqual(
                {path.name for path in evidence.iterdir()},
                {
                    f"passage_1_{camera}_frame_{offset:+d}.jpg"
                    for camera in cameras
                    for offset in range(-3, 4)
                },
            )

    def test_low_confidence_is_limited_to_loby_door_zone(self):
        directions = {
            "left_to_right": "entry",
            "right_to_left": "exit",
        }
        cameras = {
            "loby": {
                "line": ((0, 0), (0, 20)),
                "directions": directions,
                "door_confidence": 0.15,
                "door_confidence_radius_px": 5,
            },
            "entrance": {"line": ((0, 0), (0, 20)), "directions": directions},
        }
        detector = ScoredDetector([
            [((0, 0, 4, 10), 0.2)],
            [((18, 0, 4, 10), 0.2)],
            [((0, 0, 4, 10), 0.2)],
            [((0, 0, 4, 10), 0.02)],
        ])
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        started = datetime(2026, 1, 2)

        with tempfile.TemporaryDirectory() as directory:
            diagnostics = Path(directory) / "diagnostics.jsonl"
            counter = DoorCounter(
                cameras=cameras,
                model_path="unused",
                confidence=0.35,
                agreement_seconds=15,
                crossing_margin_px=1,
                database_path=Path(directory) / "people.sqlite3",
                app_version="test",
                detector=detector,
                diagnostics_path=diagnostics,
            )
            counter.update("loby", frame, started)
            counter.update("loby", frame, started + timedelta(seconds=1))
            counter.update("entrance", frame, started + timedelta(seconds=2))
            counter.update("entrance", frame, started + timedelta(seconds=3))

            self.assertEqual(len(counter.tracks["loby"]), 1)
            self.assertEqual(len(counter.tracks["entrance"]), 0)
            counter.close()
            detection = json.loads(diagnostics.read_text().splitlines()[-1])
            self.assertEqual(detection["detection_count"], 1)
            self.assertEqual(detection["detections"][0]["confidence"], 0.02)

    def test_both_cameras_record_and_confirm_entry_and_exit(self):
        cameras = {
            camera: {
                "line": ((0, 0), (0, 20)),
                "directions": {
                    "left_to_right": "entry",
                    "right_to_left": "exit",
                },
            }
            for camera in ("entrance", "loby")
        }
        boxes = [
            [(-12, 0, 4, 10)], [(8, 0, 4, 10)],
            [(-12, 0, 4, 10)], [(8, 0, 4, 10)],
            [(8, 0, 4, 10)], [(-12, 0, 4, 10)],
            [(8, 0, 4, 10)], [(-12, 0, 4, 10)],
        ]
        started = datetime(2026, 1, 2, 3, 4, 5)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "people.sqlite3"
            diagnostics = Path(directory) / "diagnostics.jsonl"
            counter = DoorCounter(
                cameras=cameras,
                model_path="unused",
                confidence=0.35,
                agreement_seconds=15,
                crossing_margin_px=1,
                database_path=database,
                app_version="0.2.1",
                detector=FakeDetector(boxes),
                diagnostics_path=diagnostics,
            )
            counter.update("entrance", frame, started)
            counter.update("entrance", frame, started + timedelta(seconds=1))
            self.assertEqual(counter.snapshot()["people_inside_confidence"], 0)
            counter.finish(started + timedelta(seconds=2))
            counter.update("loby", frame, started + timedelta(seconds=2))
            counter.update("loby", frame, started + timedelta(seconds=3))
            self.assertEqual(counter.snapshot(), {
                "entered_total": 1,
                "exited_total": 0,
                "people_inside": 1,
                "people_inside_confidence": 1.0,
            })

            counter.update("entrance", frame, started + timedelta(seconds=4))
            counter.update("entrance", frame, started + timedelta(seconds=5))
            counter.update("loby", frame, started + timedelta(seconds=6))
            counter.update("loby", frame, started + timedelta(seconds=7))

            self.assertEqual(counter.snapshot(), {
                "entered_total": 1,
                "exited_total": 1,
                "people_inside": 0,
                "people_inside_confidence": 1.0,
            })
            counter.finish(started + timedelta(seconds=20))
            counter.close()
            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT passed_at, app_version, camera, direction "
                    "FROM door_passages ORDER BY id"
                ).fetchall()
            self.assertEqual(rows, [
                ("2026-01-02 03:04:06.000", "0.2.1", "entrance", "entry"),
                ("2026-01-02 03:04:08.000", "0.2.1", "loby", "entry"),
                ("2026-01-02 03:04:10.000", "0.2.1", "entrance", "exit"),
                ("2026-01-02 03:04:12.000", "0.2.1", "loby", "exit"),
            ])
            events = {
                row["event"]
                for row in (
                    json.loads(line)
                    for line in diagnostics.read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertTrue({
                "detection", "track_update", "line_crossing",
                "passage_unconfirmed", "passage_agreement",
            }.issubset(events))
            self.assertEqual(counter.diagnostic_summary(), {
                "observations_by_camera": {"entrance": 2, "loby": 2},
                "confirmed_passages": 2,
                "unconfirmed_passages": 0,
                "direction_mismatches": 0,
            })


if __name__ == "__main__":
    unittest.main()
