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
    CONFIG_PATH, DATABASE_PATH, PROJECT_DIR,
    archive_dir as configured_archive_dir, archive_server,
    debug_mode, door_counter_settings, frame_interval_ms,
    prepare_benchmark_enabled, stream_cameras,
)
from .counter import DoorCounter
from .prepare_benchmark import prepare_benchmark
from .stream import (
    FRAME_HEIGHT, FRAME_WIDTH, FrameStore, build_archive_plan,
    common_archive_interval, remote_archive, start_decoders,
)
from .sync import PlaybackClock, archive_skew_ms, capture_needs_resync

REFERENCE_TELEMETRY = {
    "archive_debug_cache": "archive_people_telemetry.jsonl",
    "archive_short": "archive_people_telemetry_short.jsonl",
}

log = logging.getLogger(__name__)


def occupancy_exact_match_pct(telemetry, reference_path):
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
    comparable = [
        row for row in telemetry
        if row["mkv_pts_time"] in reference
    ]
    if not comparable:
        raise ValueError("telemetry does not overlap the reference")
    matches = sum(
        row["people_inside"] == reference[row["mkv_pts_time"]]
        for row in comparable
    )
    return round(matches / len(comparable) * 100, 6)


def _reference_telemetry_path(archive_dir):
    archive_name = Path(archive_dir).name
    try:
        filename = REFERENCE_TELEMETRY[archive_name]
    except KeyError as error:
        raise ValueError(
            f"no reference telemetry configured for {archive_name}"
        ) from error
    return PROJECT_DIR / "resources" / filename


def _reference_interval(reference_path, available_start, available_end):
    timestamps = [
        datetime.fromisoformat(row["mkv_pts_time"])
        for row in (
            json.loads(line)
            for line in Path(reference_path).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    ]
    timestamps.sort()
    timestamps = [
        timestamp for timestamp in timestamps
        if timestamp >= available_start
        and timestamp + timedelta(seconds=1) <= available_end
    ]
    if not timestamps:
        raise ValueError("reference telemetry does not overlap the archive")
    return timestamps[0], timestamps[-1] + timedelta(seconds=1)


def _reference_events(reference_path, start_time, end_time):
    rows = sorted(
        (
            datetime.fromisoformat(row["mkv_pts_time"]), row
        )
        for row in (
            json.loads(line)
            for line in Path(reference_path).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    )
    events = []
    previous = rows[0][1]
    for timestamp, row in rows[1:]:
        for direction, key in (
            ("entry", "entered_total"),
            ("exit", "exited_total"),
        ):
            if start_time <= timestamp < end_time:
                events.extend({
                    "id": f"{direction}_{total}",
                    "timestamp": timestamp,
                    "direction": direction,
                } for total in range(previous[key] + 1, row[key] + 1))
        previous = row
    return events


def _telemetry_record(timestamp, counter):
    return {
        "mkv_pts_time": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        **counter.snapshot(timestamp),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir", type=Path,
    )
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--start-offset-ms", type=int, default=0)
    args = parser.parse_args()
    server = archive_server()
    if args.archive_dir is not None or server is None:
        args.archive_dir = args.archive_dir or configured_archive_dir()
        return _run(args, parser)

    base_url, login, password = server
    with remote_archive(
        base_url,
        stream_cameras(),
        login,
        password,
        debug=debug_mode(),
    ) as (archive_dir, intervals, skipped_intervals):
        args.archive_dir = archive_dir
        return _run(args, parser, intervals, skipped_intervals)


def _analyze_interval(
    archive_dir, cameras, counter, start_time, end_time, interval_ms,
    *, track_reads=None,
):
    plan = build_archive_plan(
        archive_dir, start_time, end_time, cameras, interval_ms,
    )
    counts = Counter(
        (ref["camera"], ref["mkv"], ref["frame_index"])
        for refs in plan.values()
        for ref in refs
    )
    store = FrameStore(counts)
    if track_reads is None:
        threads = start_decoders(archive_dir, plan, store)
    else:
        # ponytail: rolling batches can reuse one MKV; add range checkpoints
        # if restart deduplication for remote archives becomes necessary.
        threads = start_decoders(
            archive_dir, plan, store, debug=not track_reads,
        )
    clock = PlaybackClock(start_time, interval_ms)
    decoded = 0
    max_skew = 0

    for tick in range(len(plan[cameras[0]])):
        expected_time = clock.expected_archive_time
        stats = {}
        for camera in cameras:
            ref = plan[camera][tick]
            if capture_needs_resync(
                expected_time, ref["mkv_pts_time"], clock.archive_interval,
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

    for thread in threads:
        thread.join()
    result = counter.snapshot(end_time)
    log.info(
        "analysis interval completed start=%s end=%s frames=%d ticks=%d "
        "entered=%d exited=%d people_inside=%d",
        start_time, end_time, decoded, clock.tick,
        result["entered_total"], result["exited_total"], result["people_inside"],
    )
    return decoded, clock.tick, max_skew, result


def _run(args, parser, intervals=None, skipped_intervals=None):
    if not args.archive_dir.is_dir():
        parser.error(f"archive directory not found: {args.archive_dir}")
    if args.start_offset_ms < 0:
        parser.error("start offset must not be negative")
    if args.compare_reference and args.start_offset_ms:
        parser.error("start offset cannot be used with reference comparison")

    cameras = stream_cameras()
    skipped_intervals = skipped_intervals if skipped_intervals is not None else []
    reference_path = None
    remote = intervals is not None
    if remote:
        if args.compare_reference:
            parser.error("reference comparison requires a local archive")
        reference_bounds = None
    else:
        start_time, end_time = common_archive_interval(args.archive_dir, cameras)
        if args.compare_reference:
            reference_path = _reference_telemetry_path(args.archive_dir)
            if not reference_path.is_file():
                parser.error(f"reference telemetry not found: {reference_path}")
            start_time, end_time = _reference_interval(
                reference_path, start_time, end_time,
            )
        start_time += timedelta(milliseconds=args.start_offset_ms)
        if start_time >= end_time:
            parser.error("start offset leaves no archive to process")
        reference_bounds = (start_time, end_time)
        intervals = iter((reference_bounds,))

    run_name = __version__
    if args.start_offset_ms:
        run_name += f"-offset-{args.start_offset_ms}ms"
    run_dir = PROJECT_DIR / "runs" / run_name
    telemetry_path = run_dir / "telemetry.jsonl"
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "run.log"
    config_snapshot_path = run_dir / "config.toml"
    diagnostics_path = run_dir / "diagnostics.jsonl"
    evidence_dir = run_dir / "evidence"
    motion_activity_dir = run_dir / "motion_activity"
    existing = [
        path for path in (
            telemetry_path, summary_path, log_path, config_snapshot_path,
            diagnostics_path, evidence_dir, motion_activity_dir,
        )
        if path.exists()
    ]
    if existing:
        parser.error(f"run output already exists: {existing[0]}")

    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path.write_bytes(CONFIG_PATH.read_bytes())
    command = shlex.join([
        sys.executable, "-m", "peoplein.run", *sys.argv[1:],
    ])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    if prepare_benchmark_enabled():
        prepare_benchmark()
    if debug_mode():
        DATABASE_PATH.unlink(missing_ok=True)

    counter = DoorCounter(
        **door_counter_settings(),
        database_path=DATABASE_PATH,
        app_version=__version__,
        diagnostics_path=diagnostics_path,
        evidence_dir=evidence_dir,
        motion_activity_dir=motion_activity_dir,
        reference_events=(
            _reference_events(reference_path, *reference_bounds)
            if reference_path else ()
        ),
    )
    log.info(
        "run started cameras=%s archive=%s reference=%s",
        ",".join(cameras), args.archive_dir, reference_path or "disabled",
    )
    started = time.monotonic()

    try:
        interval_ms = frame_interval_ms()
        decoded = 0
        ticks = 0
        max_skew = 0
        telemetry = []
        start_time = None
        end_time = None
        duration = 0
        offset_start = None
        processed_intervals = []
        archive_path = args.archive_dir.resolve()

        def write_summary(result, accuracy=None):
            processing_time = round(time.monotonic() - started, 3)
            summary = {
                "format_version": 1,
                "app_version": __version__,
                "run": {
                    "source_archive": str(
                        archive_path.relative_to(PROJECT_DIR)
                        if archive_path.is_relative_to(PROJECT_DIR)
                        else archive_path
                    ),
                    "start_time": start_time.isoformat(
                        timespec="milliseconds"
                    ),
                    "end_time": end_time.isoformat(timespec="milliseconds"),
                    "start_offset_ms": args.start_offset_ms,
                    "time_zone": (
                        str(start_time.tzinfo)
                        if start_time.tzinfo else None
                    ),
                    "video_duration_seconds": duration,
                    "processing_time_seconds": processing_time,
                    "skipped_intervals": skipped_intervals,
                },
                "result": {
                    "entered_total": result["entered_total"],
                    "exited_total": result["exited_total"],
                    "people_inside": result["people_inside"],
                    "passage_confirmation_ratio": result[
                        "passage_confirmation_ratio"
                    ],
                },
                **counter.diagnostic_summary(),
                "artifacts": {
                    "log": log_path.name,
                    "diagnostics": diagnostics_path.name,
                    "evidence": evidence_dir.name,
                    "motion_activity": motion_activity_dir.name,
                    "config": config_snapshot_path.name,
                },
                "command": command,
            }
            if reference_path:
                summary["run"]["reference_telemetry"] = str(
                    reference_path.relative_to(PROJECT_DIR)
                )
                if accuracy is not None:
                    summary["result"]["occupancy_exact_match_pct"] = accuracy
            temporary = summary_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(summary_path)
            log.info("summary updated end=%s", end_time)
            return processing_time

        with telemetry_path.open("x", encoding="utf-8") as output:
            for batch_start, batch_end in intervals:
                if offset_start is None:
                    offset_start = batch_start + timedelta(
                        milliseconds=args.start_offset_ms,
                    )
                batch_start = max(batch_start, offset_start)
                if batch_start >= batch_end:
                    continue
                if end_time is not None and batch_start > end_time:
                    counter.finish(end_time)
                    counter.reset_stream()
                if start_time is None:
                    start_time = batch_start
                (
                    batch_decoded, batch_ticks, batch_skew, batch_result,
                ) = _analyze_interval(
                    args.archive_dir, cameras, counter,
                    batch_start, batch_end, interval_ms,
                    track_reads=False if remote else None,
                )
                decoded += batch_decoded
                ticks += batch_ticks
                max_skew = max(max_skew, batch_skew)
                duration += (batch_end - batch_start).total_seconds()
                end_time = batch_end
                processed_intervals.append((batch_start, batch_end))
                write_summary(batch_result)

            if start_time is None:
                raise ValueError("selected camera archives do not overlap")
            counter.finish(end_time)
            for batch_start, batch_end in processed_intervals:
                timestamp = batch_start
                while timestamp < batch_end:
                    row = _telemetry_record(timestamp, counter)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    telemetry.append(row)
                    timestamp += timedelta(seconds=1)

        accuracy = (
            occupancy_exact_match_pct(telemetry, reference_path)
            if reference_path else None
        )
        processing_time = write_summary(telemetry[-1], accuracy)
        log.info(
            "run completed seconds=%d frames=%d ticks=%d max_skew_ms=%d "
            "occupancy_exact_match_pct=%s video_duration_seconds=%s "
            "processing_time_seconds=%s people_inside=%d "
            "passage_confirmation_ratio=%s",
            len(telemetry), decoded, ticks, max_skew, accuracy,
            duration, processing_time, telemetry[-1]["people_inside"],
            telemetry[-1]["passage_confirmation_ratio"],
        )
    except Exception:
        log.exception("run failed")
        raise
    finally:
        counter.close()


if __name__ == "__main__":
    main()
