"""Stream synchronized raw BGR frames from the local MKV archive."""

import argparse
import logging
import subprocess
import threading
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np

from camera_sync import PlaybackClock, archive_skew_ms, capture_needs_resync
from config import (
    DATABASE_PATH, PROJECT_DIR, debug_mode, frame_interval_ms, playback_speed,
)
from database import ensure_unread, mark_read

CAMERAS = ("entrance", "hall1", "hall2", "hall3", "hall4", "hall5", "loby")
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
SOURCE_FPS = 25
FRAME_STRIDE = 25
SEGMENT_SECONDS = 305

log = logging.getLogger(__name__)


def _mkv_files(camera_dir):
    files = []
    for path in Path(camera_dir).glob("*.mkv"):
        try:
            timestamp = datetime.strptime(path.stem, "%Y%m%d-%H%M%S")
        except ValueError:
            continue
        files.append({"name": path.name, "timestamp": timestamp})
    return sorted(files, key=lambda item: item["timestamp"])


def _file_covering(files, target):
    index = bisect_right([item["timestamp"] for item in files], target) - 1
    if index < 0:
        return None
    item = files[index]
    if target < item["timestamp"] + timedelta(seconds=SEGMENT_SECONDS):
        return item
    return None


def _video_info(path):
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_read_packets",
        "-of", "csv=p=0", str(path),
    ], text=True).strip()
    fps_text, frame_count_text = output.split(",")
    fps = float(Fraction(fps_text))
    frame_count = int(frame_count_text)
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"invalid video metadata for {path}: fps={fps}, frames={frame_count}"
        )
    return fps, frame_count


def build_archive_plan(archive_dir, start_time, end_time, interval_ms=None):
    """Build a timestamped frame plan from one shared camera clock."""
    archive_dir = Path(archive_dir)
    if end_time <= start_time:
        raise ValueError("archive end time must be after start time")

    if interval_ms is None:
        interval_ms = frame_interval_ms()
    if interval_ms <= 0:
        raise ValueError("interval_ms must be greater than zero")
    interval = interval_ms / 1000.0
    plan = {}
    for camera in CAMERAS:
        camera_dir = archive_dir / camera
        if not camera_dir.is_dir():
            raise ValueError(f"camera directory not found: {camera_dir}")
        files = _mkv_files(camera_dir)
        info = {}
        refs = []
        tick = 0
        while start_time + timedelta(seconds=tick * interval) < end_time:
            target = start_time + timedelta(seconds=tick * interval)
            item = _file_covering(files, target)
            if item is None:
                raise ValueError(f"{camera} archive does not cover {target}")
            if item["name"] not in info:
                info[item["name"]] = _video_info(camera_dir / item["name"])
            fps, frame_count = info[item["name"]]
            frame_index = min(
                round((target - item["timestamp"]).total_seconds() * fps),
                frame_count - 1,
            )
            pts_seconds = frame_index / fps
            source_time = item["timestamp"] + timedelta(seconds=pts_seconds)
            if capture_needs_resync(target, source_time, interval):
                raise RuntimeError(
                    f"{camera} cannot synchronize: expected {target}, "
                    f"source {source_time}"
                )
            refs.append({
                "camera": camera,
                "mkv": item["name"],
                "frame_index": frame_index,
                "mkv_pts_seconds": pts_seconds,
                "mkv_pts_time": source_time,
            })
            tick += 1
        plan[camera] = refs
    return plan


class FrameStore:
    """Bounded raw BGR handoff between FFmpeg decoders and a consumer."""

    def __init__(self, counts=None):
        self._counts = dict(counts) if counts is not None else None
        self._ready = {}
        self._failed = None
        self._decoders_total = None
        self._decoders_done = 0
        self._condition = threading.Condition()

    def fail(self, error):
        with self._condition:
            self._failed = error
            self._condition.notify_all()

    def track_decoders(self, count):
        with self._condition:
            self._decoders_total = count

    def decoder_done(self):
        with self._condition:
            self._decoders_done += 1
            self._condition.notify_all()

    def put(self, camera, mkv, frame_index, frame):
        with self._condition:
            while (
                sum(key[0] == camera for key in self._ready) >= 2
                and self._failed is None
            ):
                self._condition.wait(timeout=1.0)
            if self._failed is None:
                self._ready[(camera, mkv, frame_index)] = frame
                self._condition.notify_all()

    def get(self, camera, mkv, frame_index):
        key = (camera, mkv, frame_index)
        with self._condition:
            while key not in self._ready:
                if self._failed is not None:
                    raise RuntimeError(f"decoder failed: {self._failed}")
                if (
                    self._decoders_total is not None
                    and self._decoders_done >= self._decoders_total
                ):
                    raise RuntimeError(f"frame {key} was never decoded")
                self._condition.wait(timeout=1.0)
            frame = self._ready[key]
            remaining = 0
            if self._counts is not None:
                remaining = self._counts.get(key, 1) - 1
                if remaining > 0:
                    self._counts[key] = remaining
                else:
                    self._counts.pop(key, None)
                    del self._ready[key]
            else:
                del self._ready[key]
            self._condition.notify_all()
        return frame


def _decode_camera(archive_dir, camera, needed_by_file, store, database_path=None):
    frame_bytes = FRAME_WIDTH * FRAME_HEIGHT * 3
    try:
        for mkv in sorted(needed_by_file):
            needed = needed_by_file[mkv]
            if not needed:
                continue
            extras = sorted(index for index in needed if index % FRAME_STRIDE)
            terms = [f"not(mod(n,{FRAME_STRIDE}))"] + [
                f"eq(n,{index})" for index in extras
            ]
            command = [
                "ffmpeg", "-v", "error", "-threads", "1",
                "-skip_loop_filter", "all",
                "-i", str(Path(archive_dir) / camera / mkv),
                "-vf", f"select='{'+'.join(terms)}',scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
                "-fps_mode", "passthrough",
                "-frames:v", str(max(needed) // FRAME_STRIDE + 1 + len(extras)),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            emitted = _emitted_indices(FRAME_STRIDE, extras)
            remaining = set(needed)
            while True:
                chunk = _read_exactly(process.stdout, frame_bytes)
                if chunk is None:
                    break
                index = next(emitted)
                if index not in needed:
                    continue
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    FRAME_HEIGHT, FRAME_WIDTH, 3,
                )
                store.put(camera, mkv, index, frame)
                remaining.discard(index)
            stderr = process.stderr.read().decode("utf-8", "replace")
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed for {camera}/{mkv}: {stderr}")
            if remaining:
                raise RuntimeError(
                    f"{camera}/{mkv}: {len(remaining)} frames beyond EOF, "
                    f"e.g. {min(remaining)}"
                )
            if database_path is not None:
                mark_read(database_path, f"{camera}/{mkv}")
            log.info("decoded %s/%s: %d frames", camera, mkv, len(needed))
    except Exception as error:
        store.fail(error)
    finally:
        store.decoder_done()


def _emitted_indices(stride, extras):
    pending = iter(sorted(extras))
    extra = next(pending, None)
    ordinal = 0
    while True:
        index = ordinal * stride
        ordinal += 1
        while extra is not None and extra < index:
            yield extra
            extra = next(pending, None)
        yield index


def _read_exactly(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def start_decoders(
    archive_dir, plan, store, *, debug=None, database_path=DATABASE_PATH,
):
    """Start one FFmpeg decoder thread per camera."""
    if debug is None:
        debug = debug_mode()
    tracked_database = None if debug else Path(database_path)
    if tracked_database is not None:
        ensure_unread(
            tracked_database,
            (
                f"{camera}/{ref['mkv']}"
                for camera, refs in plan.items()
                for ref in refs
            ),
        )

    threads = []
    store.track_decoders(len(plan))
    for camera, refs in plan.items():
        needed_by_file = defaultdict(set)
        for ref in refs:
            needed_by_file[ref["mkv"]].add(ref["frame_index"])
        thread = threading.Thread(
            target=_decode_camera,
            args=(archive_dir, camera, needed_by_file, store, tracked_database),
            name=f"decode-{camera}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    return threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir", type=Path,
        default=PROJECT_DIR / "resources" / "archive_debug_cache",
    )
    parser.add_argument("--start-time", type=datetime.fromisoformat, required=True)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    interval_ms = frame_interval_ms()
    plan = build_archive_plan(
        args.archive_dir,
        args.start_time,
        args.start_time + timedelta(seconds=args.duration),
        interval_ms,
    )
    counts = Counter(
        (ref["camera"], ref["mkv"], ref["frame_index"])
        for refs in plan.values()
        for ref in refs
    )
    store = FrameStore(counts)
    threads = start_decoders(args.archive_dir, plan, store)
    clock = PlaybackClock(args.start_time, interval_ms, playback_speed())
    decoded = 0
    max_skew = 0
    for tick in range(len(plan[CAMERAS[0]])):
        stats = {}
        for camera in CAMERAS:
            ref = plan[camera][tick]
            if capture_needs_resync(
                clock.expected_archive_time,
                ref["mkv_pts_time"],
                clock.archive_interval,
            ):
                raise RuntimeError(
                    f"{camera} drifted from playback clock at tick {tick}"
                )
            frame = store.get(camera, ref["mkv"], ref["frame_index"])
            if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise RuntimeError(f"unexpected frame shape: {frame.shape}")
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
    print(
        f"decoded {decoded} raw BGR frames across {clock.tick} synchronized "
        f"ticks; max skew={max_skew} ms"
    )


if __name__ == "__main__":
    main()
