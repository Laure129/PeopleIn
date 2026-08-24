import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from peoplein.config import _load_archive_env, archive_server, stream_cameras
from peoplein.run import (
    _reference_events, _reference_interval, occupancy_exact_match_pct,
)
from peoplein.stream import (
    _decode_camera, common_archive_interval, remote_archive,
)
from peoplein.sync import PlaybackClock


class ArchiveRunTest(unittest.TestCase):
    def test_archive_server_loads_env_file_without_overriding_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"ARCHIVE_LOGIN": "process-user"}, clear=True,
        ):
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "ARCHIVE_BASE_URL=https://archive.test/root\n"
                "ARCHIVE_LOGIN=file-user\n"
                "ARCHIVE_PASSWORD='secret value'\n",
                encoding="utf-8",
            )

            _load_archive_env(env_path)

            self.assertEqual(archive_server(), (
                "https://archive.test/root", "process-user", "secret value",
            ))

    def test_remote_archive_cleanup_and_debug_cache(self):
        class Response(io.BytesIO):
            def __init__(self, body):
                super().__init__(body)
                self.headers = {"Content-Length": str(len(body))}

        requests = []

        def open_url(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/"):
                return Response(b'<a href="20260101-000000.mkv">video</a>')
            return Response(b"mkv")

        with tempfile.TemporaryDirectory() as directory, patch(
            "peoplein.stream.PROJECT_DIR", Path(directory),
        ), patch("peoplein.stream.urlopen", side_effect=open_url):
            with remote_archive(
                "http://archive.test/root",
                ("entrance", "loby"),
                "user",
                "secret",
            ) as archive:
                temporary_archive = archive
                self.assertEqual(
                    sorted(path.name for path in archive.glob("*/*.mkv")),
                    ["20260101-000000.mkv", "20260101-000000.mkv"],
                )
            self.assertFalse(temporary_archive.exists())

            with remote_archive(
                "http://archive.test/root",
                ("entrance", "loby"),
                "user",
                "secret",
                debug=True,
            ) as debug_archive:
                pass
            self.assertTrue(debug_archive.is_dir())
            self.assertTrue(
                debug_archive.name.startswith("archive_debug_cache_remote-")
            )

        self.assertTrue(all(
            request.get_header("Authorization") == "Basic dXNlcjpzZWNyZXQ="
            for request in requests
        ))

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

    def test_reference_events_follow_total_changes(self):
        started = datetime(2026, 1, 1)
        rows = [
            {
                "mkv_pts_time": (started + timedelta(seconds=second)).isoformat(),
                "entered_total": entered,
                "exited_total": exited,
            }
            for second, entered, exited in ((0, 0, 0), (1, 2, 0), (2, 2, 1))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            self.assertEqual(_reference_events(
                path, started, started + timedelta(seconds=3),
            ), [
                {
                    "id": "entry_1",
                    "timestamp": started + timedelta(seconds=1),
                    "direction": "entry",
                },
                {
                    "id": "entry_2",
                    "timestamp": started + timedelta(seconds=1),
                    "direction": "entry",
                },
                {
                    "id": "exit_1",
                    "timestamp": started + timedelta(seconds=2),
                    "direction": "exit",
                },
            ])

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
