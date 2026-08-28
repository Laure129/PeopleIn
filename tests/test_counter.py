import json
import sqlite3
import tempfile
import unittest
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from peoplein.counter import DoorCounter, Track


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


class RecordingDetector:
    def __init__(self):
        self.frames = []

    def __call__(self, frame):
        self.frames.append(int(frame[0, 0, 0]))
        return []


class DoorCounterTest(unittest.TestCase):
    def test_flush_accepts_duplicate_camera_timestamps(self):
        started = datetime(2026, 1, 2)
        frames = [
            np.full((1, 1, 3), value, dtype=np.uint8)
            for value in (1, 2)
        ]
        counter = DoorCounter.__new__(DoorCounter)
        counter.pending_frames = {
            "entrance": deque((frame, started, []) for frame in frames),
        }
        counter.motion = {}
        counter.door_motion_activity = []
        counter.full_camera_motion_activity = []
        counter.door_motion_activity_start = 0
        counter.full_camera_motion_activity_start = 0
        counter.tracks = {"entrance": {}}
        counter.frame_history = {"entrance": deque(maxlen=7)}
        counter._expire_events = Mock()
        counter._save_ready_evidence = Mock()

        counter._flush_frames(started, force=True)

        self.assertEqual(
            [int(frame[0, 0, 0]) for frame, _, _ in counter.frame_history["entrance"]],
            [1, 2],
        )

    def test_stream_reset_discards_only_temporal_state(self):
        counter = DoorCounter.__new__(DoorCounter)
        counter.events = [{"direction": "entry"}]
        counter.door_motion_activity = [{"camera": "loby"}]
        counter.full_camera_motion_activity = [{"camera": "loby"}]
        counter.door_motion_activity_start = 0
        counter.full_camera_motion_activity_start = 0
        counter.tracks = {"entrance": {1: object()}}
        counter.pending_frames = {"entrance": deque([object()])}
        counter.frame_history = {"entrance": deque([object()], maxlen=7)}
        counter.motion = {
            "loby": {
                "previous_gray": object(),
                "activity_gray": object(),
                "activity_samples": 3,
                "mask": object(),
            },
        }
        counter.pending_evidence = [object()]

        counter.reset_stream()

        self.assertEqual(counter.events, [{"direction": "entry"}])
        self.assertEqual(
            counter.door_motion_activity, [{"camera": "loby"}],
        )
        self.assertEqual(
            counter.full_camera_motion_activity, [{"camera": "loby"}],
        )
        self.assertEqual(counter.door_motion_activity_start, 1)
        self.assertEqual(counter.full_camera_motion_activity_start, 1)
        self.assertFalse(counter.tracks["entrance"])
        self.assertFalse(counter.pending_frames["entrance"])
        self.assertFalse(counter.frame_history["entrance"])
        self.assertIsNone(counter.motion["loby"]["previous_gray"])
        self.assertIsNone(counter.motion["loby"]["activity_gray"])
        self.assertEqual(counter.motion["loby"]["activity_samples"], 0)
        self.assertIsNotNone(counter.motion["loby"]["mask"])
        self.assertEqual(
            counter.motion_profiles,
            {"loby": counter._new_motion_profile()},
        )
        self.assertFalse(counter.pending_evidence)

    def test_motion_profile_counts_each_low_high_low_person_shape(self):
        counter = DoorCounter.__new__(DoorCounter)
        counter.cameras = {"loby": {
            "motion_min_points": 3,
            "motion_profile_open_min_points": 5,
            "motion_profile_min_points": 20,
        }}
        counter.motion_profiles = {"loby": counter._new_motion_profile()}
        counter.door_profile_entries = []
        counter.diagnostics = None
        started = datetime(2026, 1, 2)

        for tick, (entry, exit_) in enumerate((
            (0, 6),
            (25, 0), (40, 0), (0, 0),
            (22, 0), (50, 0), (0, 0),
            (24, 0), (45, 0), (0, 0),
        )):
            counter._motion_profile_bin(
                "loby", started + timedelta(milliseconds=200 * tick),
                entry, exit_,
            )

        self.assertEqual(len(counter.door_profile_entries), 3)
        self.assertEqual(
            [entry["peak_motion_points"]
             for entry in counter.door_profile_entries],
            [40, 50, 45],
        )

    def test_motion_direction_can_be_limited_to_perpendicular(self):
        geometry = {
            "line": ((0, 0), (0, 20)),
            "motion_perpendicular_only": True,
        }

        self.assertTrue(DoorCounter._motion_direction_allowed(
            (0, 0), (10, 0), geometry,
        ))
        self.assertFalse(DoorCounter._motion_direction_allowed(
            (0, 0), (0, 10), geometry,
        ))
        geometry["motion_perpendicular_only"] = False
        self.assertTrue(DoorCounter._motion_direction_allowed(
            (0, 0), (0, 10), geometry,
        ))

    def test_track_leaving_door_line_counts_as_crossing(self):
        counter = DoorCounter.__new__(DoorCounter)
        line = ((0, 0), (0, 20))

        for x, expected in (
            (5, ("left_to_right", -1)),
            (-5, ("right_to_left", 1)),
        ):
            with self.subTest(x=x):
                track = Track((0, 10), 0, (0, 10))
                self.assertEqual(
                    counter._move_track(track, (x, 10), line, margin=1),
                    expected,
                )

    def test_snapshot_uses_archive_timestamp(self):
        started = datetime(2026, 1, 2)
        counter = DoorCounter.__new__(DoorCounter)
        counter.door_profile_entries = []
        counter.events = [{
            "timestamp": started + timedelta(seconds=5, milliseconds=200),
            "direction": "entry",
            "confirmed": True,
            "confirmed_at": started + timedelta(seconds=20, milliseconds=200),
        }]
        counter.diagnostics = None

        self.assertEqual(
            counter.snapshot(started + timedelta(seconds=4))["entered_total"],
            0,
        )
        self.assertEqual(
            counter.snapshot(started + timedelta(seconds=5))["entered_total"],
            1,
        )
        self.assertEqual(
            counter.snapshot(started + timedelta(seconds=5))[
                "passage_confirmation_ratio"
            ],
            0.0,
        )

    def test_reference_evidence_saves_misses_without_duplicates(self):
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
        for tick in range(11):
            boxes.extend((
                [(-12 if tick < 7 else 8, 0, 4, 10)],
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
                reference_events=[
                    {
                        "id": "entry_1",
                        "timestamp": started + timedelta(milliseconds=600),
                        "direction": "entry",
                    },
                    {
                        "id": "entry_2",
                        "timestamp": started + timedelta(milliseconds=600),
                        "direction": "entry",
                    },
                    {
                        "id": "exit_1",
                        "timestamp": started + timedelta(milliseconds=800),
                        "direction": "exit",
                    },
                ],
            )
            for tick in range(11):
                timestamp = started + timedelta(milliseconds=200 * tick)
                for camera in cameras:
                    counter.update(camera, frame, timestamp)

            self.assertEqual(
                {path.name for path in evidence.iterdir()},
                {
                    f"passage_1_entry_{camera}_frame_{offset:+d}.jpg"
                    for camera in cameras
                    for offset in range(-3, 4)
                } | {
                    f"reference_entry_2_entry_{camera}_frame_{offset:+d}.jpg"
                    for camera in cameras
                    for offset in range(-3, 4)
                } | {
                    f"reference_exit_1_exit_{camera}_frame_{offset:+d}.jpg"
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

    def test_motion_only_confirms_entrance_passage(self):
        cameras = {
            "entrance": {
                "line": ((0, 0), (0, 80)),
                "directions": {
                    "left_to_right": "entry",
                    "right_to_left": "exit",
                },
            },
            "loby": {
                "line": ((30, 0), (30, 80)),
                "directions": {
                    "right_to_left": "entry",
                    "left_to_right": "exit",
                },
                "motion_roi": ((0, 0), (99, 79)),
                "motion_band_width_px": 40,
                "motion_min_points": 2,
                "motion_min_displacement_px": 20,
            },
        }
        started = datetime(2026, 1, 2, 3, 4, 5)
        entrance_frame = np.zeros((80, 100, 3), dtype=np.uint8)
        motion_frames = []
        texture = np.repeat(
            np.random.default_rng(1).integers(
                0, 256, (30, 30, 1), dtype=np.uint8,
            ),
            3,
            axis=2,
        )
        detections = []
        for tick in range(7):
            left = 40 if tick < 3 else 15
            frame = np.zeros_like(entrance_frame)
            frame[30:60, left:left + 30] = texture
            motion_frames.append(frame)
            detections.append([
                ((-12 if tick < 3 else 8, 0, 4, 10), 0.9),
                ((70, 60, 10, 10), 0.02),
            ])

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "people.sqlite3"
            diagnostics = Path(directory) / "diagnostics.jsonl"
            evidence = Path(directory) / "evidence"
            door_motion_activity = Path(directory) / "door_motion_activity"
            counter = DoorCounter(
                cameras=cameras,
                model_path="unused",
                confidence=0.35,
                agreement_seconds=15,
                crossing_margin_px=1,
                database_path=database,
                app_version="0.2.1",
                detector=ScoredDetector(detections),
                diagnostics_path=diagnostics,
                evidence_dir=evidence,
                door_motion_activity_dir=door_motion_activity,
                motion_profile_bin_frames=1,
            )
            self.assertIsNone(
                counter.snapshot()["passage_confirmation_ratio"]
            )
            for tick in range(7):
                timestamp = started + timedelta(seconds=tick)
                counter.update("entrance", entrance_frame, timestamp)
                counter.update("loby", motion_frames[tick], timestamp)
            self.assertEqual(counter.diagnostic_summary()["passages"], [])
            counter.finish(started + timedelta(seconds=20))
            self.assertEqual(counter.snapshot(), {
                "entered_total": 1,
                "exited_total": 0,
                "people_inside": 1,
                "door_profile_entered_total": 0,
                "passage_confirmation_ratio": 1.0,
            })
            self.assertEqual(
                {path.name for path in evidence.iterdir()},
                {
                    f"passage_1_entry_{camera}_frame_{offset:+d}.jpg"
                    for camera in cameras for offset in range(-3, 4)
                },
            )
            self.assertEqual(
                len(list(door_motion_activity.glob("*.jpg"))), 1,
            )
            image = cv2.imread(str(
                evidence / "passage_1_entry_entrance_frame_+0.jpg"
            ))
            self.assertLess(int(image[60, 70].max()), 80)
            image = cv2.imread(str(
                evidence / "passage_1_entry_loby_frame_+0.jpg"
            ))
            self.assertTrue(np.any(
                (image[:, :, 1] > 150)
                & (image[:, :, 0] > 150)
                & (image[:, :, 2] < 100)
            ))

            counter.close()
            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT passed_at, app_version, camera, direction "
                    "FROM door_passages ORDER BY id"
                ).fetchall()
            self.assertEqual(rows, [
                ("2026-01-02 03:04:08.000", "0.2.1", "entrance", "entry"),
            ])
            events = {
                row["event"]
                for row in (
                    json.loads(line)
                    for line in diagnostics.read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertTrue({
                "motion_flow", "detection", "track_update",
                "line_crossing", "passage_agreement",
            }.issubset(events))
            self.assertEqual(counter.diagnostic_summary(), {
                "door_motion_activity_intervals": [{
                    "camera": "loby",
                    "start": "2026-01-02T03:04:08.000",
                    "end": "2026-01-02T03:04:08.000",
                }],
                "full_camera_motion_activity_intervals": [{
                    "camera": "loby",
                    "start": "2026-01-02T03:04:08.000",
                    "end": "2026-01-02T03:04:08.000",
                }],
                "passages": [{
                    "timestamp": "2026-01-02T03:04:08.000",
                    "direction": "entry",
                    "camera": "entrance",
                    "confirmation": {
                        "activity": "door_motion_activity",
                        "camera": "loby",
                        "timestamp": "2026-01-02T03:04:08.000",
                        "delta_seconds": 0.0,
                    },
                }],
            })

    def test_people_are_analyzed_only_within_five_seconds_of_motion(self):
        directions = {
            "left_to_right": "entry",
            "right_to_left": "exit",
        }
        cameras = {
            "entrance": {
                "line": ((0, 0), (0, 20)),
                "directions": directions,
            },
            "loby": {
                "line": ((0, 0), (0, 20)),
                "directions": directions,
                "motion_roi": ((0, 0), (19, 19)),
                "motion_band_width_px": 5,
                "motion_min_points": 1,
                "motion_min_displacement_px": 1,
            },
        }
        detector = RecordingDetector()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        started = datetime(2026, 1, 2)

        with tempfile.TemporaryDirectory() as directory:
            counter = DoorCounter(
                cameras=cameras,
                model_path="unused",
                confidence=0.35,
                agreement_seconds=15,
                crossing_margin_px=1,
                database_path=Path(directory) / "people.sqlite3",
                app_version="test",
                detector=detector,
            )
            counter._motion_flow = lambda _camera, image: (
                (0, [], 1, {"entry": 0, "exit": 0})
                if image[0, 0, 0] == 6
                else (0, [], 0, {"entry": 0, "exit": 0})
            )
            for second in range(13):
                timestamp = started + timedelta(seconds=second)
                marked = np.full_like(frame, second)
                counter.update("entrance", marked, timestamp)
                counter.update("loby", marked, timestamp)
            counter.finish(started + timedelta(seconds=13))

        self.assertEqual(detector.frames, list(range(1, 12)))
        self.assertEqual(counter.diagnostic_summary(), {
            "door_motion_activity_intervals": [],
            "full_camera_motion_activity_intervals": [{
                "camera": "loby",
                "start": "2026-01-02T00:00:06.000",
                "end": "2026-01-02T00:00:06.000",
            }],
            "passages": [],
        })

    def test_door_motion_confirms_before_full_camera_motion(self):
        started = datetime(2026, 1, 2)
        counter = DoorCounter.__new__(DoorCounter)
        counter.events = [
            {
                "timestamp": timestamp,
                "direction": "entry",
                "observations": [{
                    "id": index,
                    "timestamp": timestamp,
                    "camera": "entrance",
                }],
                "confirmed": False,
            }
            for index, timestamp in enumerate(
                (started, started + timedelta(seconds=100)), start=1,
            )
        ]
        counter.agreement_seconds = 15
        counter.door_motion_activity = [{
            "timestamp": started + timedelta(seconds=10),
            "camera": "loby",
            "motion_points": 1,
            "activity": "door_motion_activity",
        }]
        counter.full_camera_motion_activity = [
            {
                "timestamp": timestamp,
                "camera": "loby",
                "motion_points": 1,
                "activity": "full_camera_motion_activity",
            }
            for timestamp in (started, started + timedelta(seconds=100))
        ]
        counter.door_motion_activity_start = 0
        counter.full_camera_motion_activity_start = 0
        counter.diagnostics = None

        counter._match_motion(started + timedelta(seconds=200), force=True)

        self.assertEqual(
            [event["motion_candidate"]["activity"] for event in counter.events],
            ["door_motion_activity", "full_camera_motion_activity"],
        )

    def test_activity_merges_only_gaps_under_three_seconds(self):
        started = datetime(2026, 1, 2, 3, 4, 5)
        counter = DoorCounter.__new__(DoorCounter)
        counter.motion = {"loby": {}}
        activities = [
            {"camera": "loby", "timestamp": started + timedelta(seconds=gap)}
            for gap in (0, 2, 5, 7)
        ]

        self.assertEqual(counter._activity_intervals(activities), [
            {
                "camera": "loby",
                "start": "2026-01-02T03:04:05.000",
                "end": "2026-01-02T03:04:07.000",
            },
            {
                "camera": "loby",
                "start": "2026-01-02T03:04:10.000",
                "end": "2026-01-02T03:04:12.000",
            },
        ])


if __name__ == "__main__":
    unittest.main()
