import tempfile
import unittest
from pathlib import Path

from peoplein.config import prepare_benchmark_enabled
from peoplein.prepare_benchmark import Process, select_targets


def process(pid, sid, uid, comm, ticks):
    return Process(pid, 1, sid, uid, "S", comm, ticks, pid * 100)


class PrepareBenchmarkTest(unittest.TestCase):
    def test_config_defaults_off_but_project_enables_it(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.toml"
            self.assertFalse(prepare_benchmark_enabled(missing))
        self.assertTrue(prepare_benchmark_enabled())

    def test_select_targets_protects_caller_and_blocks_other_users(self):
        before = {
            1: process(1, 10, 1000, "codex", 0),
            2: process(2, 20, 1000, "codex", 0),
            3: process(3, 30, 1000, "python", 0),
            4: process(4, 40, 2000, "kimi-code", 0),
            5: process(5, 50, 1000, "python", 0),
        }
        after = {
            1: process(1, 10, 1000, "codex", 50),
            2: process(2, 20, 1000, "codex", 1),
            3: process(3, 30, 1000, "python", 20),
            4: process(4, 40, 2000, "kimi-code", 1),
            5: process(5, 50, 1000, "python", 9),
        }

        targets, blocked = select_targets(before, after, {10}, 1000, 1.0, 100)

        self.assertEqual(targets, {2, 3})
        self.assertEqual(blocked, {4})


if __name__ == "__main__":
    unittest.main()
