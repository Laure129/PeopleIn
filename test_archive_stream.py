import json
import tempfile
import unittest
from pathlib import Path

from archive_stream import analyze_second, people_inside_match_pct
from config import stream_cameras


class ArchiveRunTest(unittest.TestCase):
    def test_placeholder_analysis_and_accuracy(self):
        self.assertEqual(stream_cameras(), ("entrance", "loby"))
        self.assertEqual(
            analyze_second({"entrance": [], "loby": []}),
            {
                "entered_total": 0,
                "exited_total": 0,
                "people_inside": 0,
                "candidate_total": 0,
            },
        )

        reference = [
            {"mkv_pts_time": "2026-01-01 00:00:00.000", "people_inside": 0},
            {"mkv_pts_time": "2026-01-01 00:00:01.000", "people_inside": 1},
        ]
        telemetry = [
            {"mkv_pts_time": "2026-01-01 00:00:00.000", "people_inside": 0},
            {"mkv_pts_time": "2026-01-01 00:00:01.000", "people_inside": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in reference),
                encoding="utf-8",
            )
            self.assertEqual(people_inside_match_pct(telemetry, path), 50.0)


if __name__ == "__main__":
    unittest.main()
