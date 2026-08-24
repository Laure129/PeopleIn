import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from peoplein.config import stream_cameras
from peoplein.run import _reference_interval, occupancy_exact_match_pct
from peoplein.stream import _decode_camera, common_archive_interval
from peoplein.sync import PlaybackClock


class ArchiveRunTest(unittest.TestCase):
    @patch("peoplein.stream._video_info")
    def test_common_archive_and_reference_interval(self, video_info):
        video_info.side_effect = [
            (1, 10, 0),
            (1, 4, 2),
            (1, 2, 0),
        ]
        reference = [
            {"mkv_pts_time": "2026-01-01 00:00:02.000"},
            {"mkv_pts_time": "2026-01-01 00:00:03.000"},
            {"mkv_pts_time": "2026-01-01 00:00:05.000"},
            {"mkv_pts_time": "2026-01-01 00:00:06.000"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "archive"
            entrance = archive / "entrance"
            entrance.mkdir(parents=True)
            (entrance / "20260101-000000.mkv").touch()
            loby = archive / "loby"
            loby.mkdir()
            (loby / "20260101-000000.mkv").touch()
            (loby / "20260101-000007.mkv").touch()
            reference_path = Path(directory) / "reference.jsonl"
            reference_path.write_text(
                "".join(json.dumps(row) + "\n" for row in reference),
                encoding="utf-8",
            )

            available = common_archive_interval(
                archive, ("entrance", "loby"),
            )

            self.assertEqual(available, (
                datetime(2026, 1, 1, 0, 0, 2),
                datetime(2026, 1, 1, 0, 0, 6),
            ))
            self.assertEqual(
                _reference_interval(reference_path, *available),
                (
                    datetime(2026, 1, 1, 0, 0, 2),
                    datetime(2026, 1, 1, 0, 0, 6),
                ),
            )

    def test_archive_clock_advances_without_playback_settings(self):
        started = datetime(2026, 1, 1)
        clock = PlaybackClock(started, 333)

        clock.advance()

        self.assertEqual(clock.tick, 1)
        self.assertEqual(
            clock.expected_archive_time,
            started + timedelta(milliseconds=333),
        )

    def test_camera_config_and_accuracy(self):
        self.assertEqual(stream_cameras(), ("entrance", "loby"))

        reference = [
            {"mkv_pts_time": "2026-01-01 00:00:00.000", "people_inside": 0},
            {"mkv_pts_time": "2026-01-01 00:00:01.000", "people_inside": 1},
        ]
        telemetry = [
            {"mkv_pts_time": "2026-01-01 00:00:00.000", "people_inside": 0},
            {"mkv_pts_time": "2026-01-01 00:00:01.000", "people_inside": 0},
            {"mkv_pts_time": "2026-01-01 00:00:02.000", "people_inside": 9},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in reference),
                encoding="utf-8",
            )
            self.assertEqual(occupancy_exact_match_pct(telemetry, path), 50.0)

    @patch("peoplein.stream.FRAME_WIDTH", 1)
    @patch("peoplein.stream.FRAME_HEIGHT", 1)
    @patch("peoplein.stream.subprocess.Popen")
    def test_decoder_command_stays_small_for_long_run(self, popen):
        process = popen.return_value
        process.stdout = io.BytesIO(b"\0" * 3000)
        process.stderr = io.BytesIO()
        process.wait.return_value = 0
        store = Mock()

        _decode_camera("archive", "entrance", {"video.mkv": {0, 333, 999}}, store)

        command = popen.call_args.args[0]
        self.assertNotIn("select=", " ".join(command))
        self.assertEqual(command[command.index("-frames:v") + 1], "1000")
        self.assertEqual(
            [call.args[2] for call in store.put.call_args_list],
            [0, 333, 999],
        )


if __name__ == "__main__":
    unittest.main()
