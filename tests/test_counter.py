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
        return next(self.boxes)


class DoorCounterTest(unittest.TestCase):
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
            counter = DoorCounter(
                cameras=cameras,
                model_path="unused",
                confidence=0.35,
                agreement_seconds=15,
                crossing_margin_px=1,
                database_path=database,
                app_version="0.2.0",
                detector=FakeDetector(boxes),
            )
            counter.update("entrance", frame, started)
            counter.update("entrance", frame, started + timedelta(seconds=1))
            self.assertEqual(counter.snapshot()["people_inside_confidence"], 0)
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
            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT passed_at, app_version, camera, direction "
                    "FROM door_passages ORDER BY id"
                ).fetchall()
            self.assertEqual(rows, [
                ("2026-01-02 03:04:06.000", "0.2.0", "entrance", "entry"),
                ("2026-01-02 03:04:08.000", "0.2.0", "loby", "entry"),
                ("2026-01-02 03:04:10.000", "0.2.0", "entrance", "exit"),
                ("2026-01-02 03:04:12.000", "0.2.0", "loby", "exit"),
            ])


if __name__ == "__main__":
    unittest.main()
