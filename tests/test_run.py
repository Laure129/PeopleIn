import io
import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from peoplein.config import _load_archive_env, archive_server, stream_cameras
from peoplein.run import (
    _reference_events, _reference_interval, _run, occupancy_exact_match_pct,
)
from peoplein.stream import (
    _decode_camera, _remote_intervals, common_archive_interval, remote_archive,
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

    @patch("peoplein.stream._video_info")
    def test_remote_archive_streams_misaligned_files(self, _video_info):
        class Response(io.BytesIO):
            def __init__(self, body):
                super().__init__(body)
                self.headers = {"Content-Length": str(len(body))}

        requests = []

        def video_info(path):
            if path.read_bytes() != b"mkv":
                raise RuntimeError("broken cache")
            return 1, 5, 0

        _video_info.side_effect = video_info

        def open_url(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/"):
                start = "000000" if "/entrance/" in request.full_url else "000002"
                second = "000005" if start == "000000" else "000007"
                return Response((
                    f'<a href="20260101-{start}.mkv">video</a>'
                    f'<a href="20260101-{second}.mkv">video</a>'
                ).encode())
            return Response(b"mkv")

        with tempfile.TemporaryDirectory() as directory, patch(
            "peoplein.stream.PROJECT_DIR", Path(directory),
        ), patch("peoplein.stream.urlopen", side_effect=open_url):
            with remote_archive(
                "http://archive.test/root",
                ("entrance", "loby"),
                "user",
                "secret",
            ) as (archive, intervals, skipped):
                temporary_archive = archive
                self.assertEqual(
                    list(intervals),
                    [
                        (
                            datetime(2026, 1, 1, 0, 0, 2),
                            datetime(2026, 1, 1, 0, 0, 5),
                        ),
                        (
                            datetime(2026, 1, 1, 0, 0, 5),
                            datetime(2026, 1, 1, 0, 0, 7),
                        ),
                        (
                            datetime(2026, 1, 1, 0, 0, 7),
                            datetime(2026, 1, 1, 0, 0, 10),
                        ),
                    ],
                )
                self.assertEqual(
                    [(row["start"], row["end"]) for row in skipped],
                    [
                        ("2026-01-01T00:00:00.000", "2026-01-01T00:00:02.000"),
                        ("2026-01-01T00:00:10.000", "2026-01-01T00:00:12.000"),
                    ],
                )
            self.assertFalse(temporary_archive.exists())

            with remote_archive(
                "http://archive.test/root",
                ("entrance", "loby"),
                "user",
                "secret",
                debug=True,
            ) as (debug_archive, intervals, _skipped):
                list(intervals)
            downloads_before_reuse = sum(
                not request.full_url.endswith("/") for request in requests
            )
            with remote_archive(
                "http://archive.test/root",
                ("entrance", "loby"),
                "user",
                "secret",
                debug=True,
            ) as (reused_archive, intervals, _skipped):
                list(intervals)
            self.assertTrue(debug_archive.is_dir())
            self.assertEqual(
                debug_archive,
                Path(directory) / "resources" / "archive_debug_cache_remote",
            )
            self.assertEqual(reused_archive, debug_archive)
            self.assertEqual(len(list(debug_archive.glob("*/*.mkv"))), 4)
            self.assertEqual(sum(
                not request.full_url.endswith("/") for request in requests
            ), downloads_before_reuse)
            corrupted = next(debug_archive.glob("*/*.mkv"))
            corrupted.write_bytes(b"broken")
            with remote_archive(
                "http://archive.test/root",
                ("entrance", "loby"),
                "user",
                "secret",
                debug=True,
            ) as (_archive, intervals, _skipped):
                list(intervals)
            self.assertEqual(corrupted.read_bytes(), b"mkv")
            self.assertEqual(sum(
                not request.full_url.endswith("/") for request in requests
            ), downloads_before_reuse + 1)

        self.assertTrue(all(
            request.get_header("Authorization") == "Basic dXNlcjpzZWNyZXQ="
            for request in requests
        ))

    def test_remote_archive_interval_edge_cases(self):
        started = datetime(2026, 1, 1)
        cases = {
            "tail_joins_next_file": (
                {"entrance": [(0, 10)], "loby": [(0, 5), (5, 5)]},
                [(0, 5), (5, 10)],
                [],
            ),
            "shifted_end": (
                {"entrance": [(0, 10)], "loby": [(2, 5)]},
                [(2, 7)],
                [(0, 2), (7, 10)],
            ),
            "short_video": (
                {"entrance": [(0, 0.2)], "loby": [(0, 5)]},
                [(0, 0.2)],
                [(0.2, 5)],
            ),
            "gap_one_camera": (
                {"entrance": [(0, 5), (20, 5)], "loby": [(0, 25)]},
                [(0, 5), (20, 25)],
                [(5, 20)],
            ),
            "gap_both_cameras": (
                {
                    "entrance": [(0, 5), (20, 5)],
                    "loby": [(0, 5), (20, 5)],
                },
                [(0, 5), (20, 25)],
                [(5, 20)],
            ),
            "large_gap_one_camera": (
                {
                    "entrance": [(0, 5), (3600, 5)],
                    "loby": [(0, 3605)],
                },
                [(0, 5), (3600, 3605)],
                [(5, 3600)],
            ),
            "large_gap_both_cameras": (
                {
                    "entrance": [(0, 5), (3600, 5)],
                    "loby": [(0, 5), (3600, 5)],
                },
                [(0, 5), (3600, 3605)],
                [(5, 3600)],
            ),
            "no_overlap": (
                {"entrance": [(0, 5)], "loby": [(20, 5)]},
                [],
                [(0, 25)],
            ),
        }

        for label, (files, expected, expected_skipped) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                names = {
                    camera: [
                        (started + timedelta(seconds=start)).strftime(
                            "%Y%m%d-%H%M%S.mkv"
                        )
                        for start, _ in videos
                    ]
                    for camera, videos in files.items()
                }
                durations = {
                    (camera, name): duration
                    for camera, videos in files.items()
                    for name, (_, duration) in zip(names[camera], videos)
                }

                def remote_names(url, _login, _password):
                    return names[url.rstrip("/").rsplit("/", 1)[-1]]

                def download(_url, destination, _login, _password):
                    destination.write_bytes(b"mkv")

                def video_info(path):
                    duration = durations[(path.parent.name, path.name)]
                    return 10, round(duration * 10), 0

                skipped = []
                with patch(
                    "peoplein.stream._remote_mkv_names", side_effect=remote_names,
                ), patch(
                    "peoplein.stream._download_mkv", side_effect=download,
                ), patch(
                    "peoplein.stream._video_info", side_effect=video_info,
                ):
                    actual = list(_remote_intervals(
                        "http://archive.test", tuple(files), Path(directory),
                        "", "", skipped,
                    ))

                self.assertEqual(actual, [
                    (
                        started + timedelta(seconds=start),
                        started + timedelta(seconds=end),
                    )
                    for start, end in expected
                ])
                self.assertEqual([
                    (
                        datetime.fromisoformat(row["start"]),
                        datetime.fromisoformat(row["end"]),
                    )
                    for row in skipped
                ], [
                    (
                        started + timedelta(seconds=start),
                        started + timedelta(seconds=end),
                    )
                    for start, end in expected_skipped
                ])
                self.assertTrue(all(
                    row["reason"] == "camera_archives_do_not_overlap"
                    for row in skipped
                ))
                self.assertEqual(
                    [row["duration_seconds"] for row in skipped],
                    [end - start for start, end in expected_skipped],
                )

    @patch("peoplein.stream._video_info", return_value=(1, 5, 0))
    @patch(
        "peoplein.stream._remote_mkv_names",
        return_value=["20260101-000000.mkv"],
    )
    def test_remote_archive_starts_camera_downloads_together(
        self, _remote_names, _video_info,
    ):
        barrier = threading.Barrier(2)

        def download(_url, destination, _login, _password):
            barrier.wait(timeout=1)
            destination.write_bytes(b"mkv")

        with tempfile.TemporaryDirectory() as directory, patch(
            "peoplein.stream._download_mkv", side_effect=download,
        ):
            self.assertEqual(list(_remote_intervals(
                "http://archive.test", ("entrance", "loby"),
                Path(directory), "", "",
            )), [
                (
                    datetime(2026, 1, 1),
                    datetime(2026, 1, 1, 0, 0, 5),
                ),
            ])

    def test_remote_run_omits_gap_telemetry_and_summarizes_it(self):
        started = datetime(2026, 1, 1)
        intervals = iter((
            (started, started + timedelta(seconds=2)),
            (
                started + timedelta(seconds=10),
                started + timedelta(seconds=12),
            ),
        ))
        skipped = [{
            "start": "2026-01-01T00:00:02.000",
            "end": "2026-01-01T00:00:10.000",
            "duration_seconds": 8.0,
            "reason": "camera_archives_do_not_overlap",
        }]

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            archive = project / "archive"
            archive.mkdir()
            config = project / "config.toml"
            config.write_text("debug = false\n", encoding="utf-8")
            args = SimpleNamespace(
                archive_dir=archive,
                compare_reference=False,
                start_offset_ms=0,
            )
            counter = Mock()
            counter.snapshot.return_value = {
                "entered_total": 1,
                "exited_total": 0,
                "people_inside": 1,
                "passage_confirmation_ratio": 1.0,
            }
            counter.diagnostic_summary.return_value = {}

            with patch("peoplein.run.PROJECT_DIR", project), patch(
                "peoplein.run.CONFIG_PATH", config,
            ), patch(
                "peoplein.run.DATABASE_PATH", project / "read.sqlite3",
            ), patch(
                "peoplein.run.__version__", "test",
            ), patch(
                "peoplein.run.prepare_benchmark_enabled", return_value=False,
            ), patch(
                "peoplein.run.debug_mode", return_value=False,
            ), patch(
                "peoplein.run.door_counter_settings", return_value={},
            ), patch(
                "peoplein.run.DoorCounter", return_value=counter,
            ), patch(
                "peoplein.run._analyze_interval", return_value=(2, 1, 0),
            ), patch("peoplein.run.logging.basicConfig"):
                _run(args, Mock(), intervals, skipped)

            run_dir = project / "runs" / "test"
            telemetry = [
                json.loads(line)
                for line in (run_dir / "telemetry.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["mkv_pts_time"] for row in telemetry],
                [
                    "2026-01-01 00:00:00.000",
                    "2026-01-01 00:00:01.000",
                    "2026-01-01 00:00:10.000",
                    "2026-01-01 00:00:11.000",
                ],
            )
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["run"]["skipped_intervals"], skipped)
            counter.reset_stream.assert_called_once_with()

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
