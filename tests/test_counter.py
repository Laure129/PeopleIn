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
            diagnostics = Path(directory) / "diagnostics.jsonl"
            evidence = Path(directory) / "evidence"
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
                evidence_dir=evidence,
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
            self.assertTrue((evidence / "passage_1_entrance.jpg").is_file())
            self.assertEqual(counter.diagnostic_summary(), {
                "observations_by_camera": {"entrance": 2, "loby": 2},
                "confirmed_passages": 2,
                "unconfirmed_passages": 0,
                "direction_mismatches": 0,
            })


if __name__ == "__main__":
    unittest.main()
