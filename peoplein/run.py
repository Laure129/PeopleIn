"""Run entrance/exit archive analysis and write versioned artifacts."""

import argparse
import json
import logging
import shlex
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from . import __version__
from .config import (
    DATABASE_PATH, PROJECT_DIR, archive_dir as configured_archive_dir,
    debug_mode, door_counter_settings, frame_interval_ms,
    prepare_benchmark_enabled, stream_cameras,
)
from .counter import DoorCounter
from .prepare_benchmark import prepare_benchmark
from .stream import (
    FRAME_HEIGHT, FRAME_WIDTH, FrameStore, build_archive_plan, start_decoders,
)
from .sync import PlaybackClock, archive_skew_ms, capture_needs_resync

REFERENCE_TELEMETRY = {
    "archive_debug_cache": "archive_people_telemetry.jsonl",
    "archive_short": "archive_people_telemetry_short.jsonl",
}

log = logging.getLogger(__name__)


def people_inside_match_pct(telemetry, reference_path):
    if not telemetry:
        raise ValueError("telemetry is empty")
    reference = {
        row["mkv_pts_time"]: row["people_inside"]
        for row in (
            json.loads(line)
            for line in Path(reference_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    missing = [
        row["mkv_pts_time"]
        for row in telemetry
        if row["mkv_pts_time"] not in reference
    ]
    if missing:
        raise ValueError(f"reference telemetry has no timestamp {missing[0]}")
    matches = sum(
        row["people_inside"] == reference[row["mkv_pts_time"]]
        for row in telemetry
    )
    return round(matches / len(telemetry) * 100, 6)


def _reference_telemetry_path(archive_dir):
    archive_name = Path(archive_dir).name
    try:
        filename = REFERENCE_TELEMETRY[archive_name]
    except KeyError as error:
        raise ValueError(
            f"no reference telemetry configured for {archive_name}"
        ) from error
    return PROJECT_DIR / "resources" / filename


def _telemetry_record(timestamp, counter):
    return {
        "mkv_pts_time": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        **counter.snapshot(timestamp),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir", type=Path,
        default=configured_archive_dir(),
    )
    parser.add_argument("--start-time", type=datetime.fromisoformat, required=True)
    parser.add_argument("--duration", type=int, default=1)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if not args.archive_dir.is_dir():
        parser.error(f"archive directory not found: {args.archive_dir}")

    cameras = stream_cameras()
    reference_path = _reference_telemetry_path(args.archive_dir)
    if not reference_path.is_file():
        parser.error(f"reference telemetry not found: {reference_path}")

    run_dir = PROJECT_DIR / "runs" / __version__
    telemetry_path = run_dir / "telemetry.jsonl"
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "run.log"
    command_path = run_dir / "command.txt"
    diagnostics_path = run_dir / "diagnostics.jsonl"
    evidence_dir = run_dir / "evidence"
    existing = [
        path for path in (
            telemetry_path, summary_path, log_path, command_path,
            diagnostics_path, evidence_dir,
        )
        if path.exists()
    ]
    if existing:
        parser.error(f"run output already exists: {existing[0]}")

    if prepare_benchmark_enabled():
        prepare_benchmark()
    if debug_mode():
        DATABASE_PATH.unlink(missing_ok=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    counter = DoorCounter(
        **door_counter_settings(),
        database_path=DATABASE_PATH,
        app_version=__version__,
        diagnostics_path=diagnostics_path,
        evidence_dir=evidence_dir,
    )
    command = shlex.join([
        sys.executable, "-m", "peoplein.run", *sys.argv[1:],
    ])
    command_path.write_text(command + "\n", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log.info(
        "run started cameras=%s archive=%s reference=%s",
        ",".join(cameras), args.archive_dir, reference_path,
    )
    started = time.monotonic()

    try:
        interval_ms = frame_interval_ms()
        plan = build_archive_plan(
            args.archive_dir,
            args.start_time,
            args.start_time + timedelta(seconds=args.duration),
            cameras,
            interval_ms,
        )
        counts = Counter(
            (ref["camera"], ref["mkv"], ref["frame_index"])
            for refs in plan.values()
            for ref in refs
        )
        store = FrameStore(counts)
        threads = start_decoders(args.archive_dir, plan, store)
        clock = PlaybackClock(args.start_time, interval_ms)
        decoded = 0
        max_skew = 0
        current_second = 0
        telemetry = []

        with telemetry_path.open("x", encoding="utf-8") as output:
            for tick in range(len(plan[cameras[0]])):
                expected_time = clock.expected_archive_time
                second = int((expected_time - args.start_time).total_seconds())
                if second != current_second:
                    row = _telemetry_record(
                        args.start_time + timedelta(seconds=current_second),
                        counter,
                    )
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    telemetry.append(row)
                    current_second = second

                stats = {}
                for camera in cameras:
                    ref = plan[camera][tick]
                    if capture_needs_resync(
                        expected_time,
                        ref["mkv_pts_time"],
                        clock.archive_interval,
                    ):
                        raise RuntimeError(
                            f"{camera} drifted from playback clock at tick {tick}"
                        )
                    frame = store.get(camera, ref["mkv"], ref["frame_index"])
                    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                        raise RuntimeError(f"unexpected frame shape: {frame.shape}")
                    counter.update(camera, frame, ref["mkv_pts_time"])
                    stats[camera] = {
                        "playback_tick": tick,
                        "mkv_pts_timestamp": ref["mkv_pts_time"].timestamp(),
                    }
                    decoded += 1
                skew = archive_skew_ms(stats) or 0
                if skew > interval_ms:
                    raise RuntimeError(f"camera skew is {skew} ms at tick {tick}")
                max_skew = max(max_skew, skew)
                clock.advance()

            counter.finish(args.start_time + timedelta(seconds=args.duration))
            row = _telemetry_record(
                args.start_time + timedelta(seconds=current_second), counter,
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            telemetry.append(row)

        for thread in threads:
            thread.join()

        accuracy = people_inside_match_pct(telemetry, reference_path)
        processing_time = round(time.monotonic() - started, 3)
        summary = {
            "video_duration_seconds": args.duration,
            "processing_time_seconds": processing_time,
            "people_inside_match_pct": accuracy,
            "entered_total": telemetry[-1]["entered_total"],
            "exited_total": telemetry[-1]["exited_total"],
            "people_inside": telemetry[-1]["people_inside"],
            "people_inside_confidence": telemetry[-1][
                "people_inside_confidence"
            ],
            **counter.diagnostic_summary(),
            "log_file": str(log_path.relative_to(PROJECT_DIR)),
            "diagnostics_file": str(diagnostics_path.relative_to(PROJECT_DIR)),
            "evidence_dir": str(evidence_dir.relative_to(PROJECT_DIR)),
            "command": command,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log.info(
            "run completed seconds=%d frames=%d ticks=%d max_skew_ms=%d "
            "people_inside_match_pct=%s video_duration_seconds=%d "
            "processing_time_seconds=%s people_inside=%d confidence=%s",
            len(telemetry), decoded, clock.tick, max_skew, accuracy,
            args.duration, processing_time, telemetry[-1]["people_inside"],
            telemetry[-1]["people_inside_confidence"],
        )
    except Exception:
        log.exception("run failed")
        raise
    finally:
        counter.close()


if __name__ == "__main__":
    main()
