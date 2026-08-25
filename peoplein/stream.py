"""Decode synchronized raw BGR frames from a local MKV archive."""

import base64
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np

from .config import (
    DATABASE_PATH, PROJECT_DIR, debug_mode, frame_interval_ms,
)
from .database import ensure_unread, mark_read
from .sync import capture_needs_resync

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
SEGMENT_SECONDS = 305
log = logging.getLogger(__name__)
_HREF_RE = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\']', re.IGNORECASE)


def _archive_request(url, login, password):
    headers = {"User-Agent": "PeopleIn/1.0"}
    if login or password:
        token = base64.b64encode(
            f"{login}:{password}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    return Request(url, headers=headers)


def _remote_mkv_names(camera_url, login, password):
    with urlopen(
        _archive_request(camera_url, login, password), timeout=15,
    ) as response:
        html = response.read().decode("utf-8", "replace")
    names = set()
    for href in _HREF_RE.findall(html):
        name = unquote(urlparse(href).path.rstrip("/").rsplit("/", 1)[-1])
        try:
            datetime.strptime(Path(name).stem, "%Y%m%d-%H%M%S")
        except ValueError:
            continue
        if Path(name).name == name and name.endswith(".mkv"):
            names.add(name)
    return sorted(names)


def _download_mkv(url, destination, login, password):
    temporary = destination.with_suffix(".mkv.part")
    try:
        with urlopen(
            _archive_request(url, login, password), timeout=60,
        ) as response, temporary.open("wb") as output:
            expected_size = response.headers.get("Content-Length")
            shutil.copyfileobj(response, output, length=1024 * 1024)
        size = temporary.stat().st_size
        if not size or expected_size and size != int(expected_size):
            raise IOError(
                f"incomplete download {destination.name}: "
                f"expected {expected_size or 'non-empty'}, got {size} bytes"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def remote_archive(base_url, cameras, login="", password="", debug=False):
    """Yield a rolling local cache and its synchronized remote intervals."""
    temporary = None
    if debug:
        resources = PROJECT_DIR / "resources"
        resources.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_dir = Path(tempfile.mkdtemp(
            prefix=f"archive_debug_cache_remote-{stamp}-",
            dir=resources,
        ))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="peoplein-archive-")
        archive_dir = Path(temporary.name)

    skipped_intervals = []
    try:
        yield archive_dir, _remote_intervals(
            base_url, cameras, archive_dir, login, password,
            skipped_intervals,
        ), skipped_intervals
    finally:
        if temporary is not None:
            temporary.cleanup()


def _record_skipped_interval(skipped_intervals, start, end):
    if start >= end:
        return
    interval = {
        "start": start.isoformat(timespec="milliseconds"),
        "end": end.isoformat(timespec="milliseconds"),
        "duration_seconds": (end - start).total_seconds(),
        "reason": "camera_archives_do_not_overlap",
    }
    skipped_intervals.append(interval)
    log.warning(
        "archive interval skipped start=%s end=%s duration_seconds=%s "
        "reason=%s",
        interval["start"], interval["end"], interval["duration_seconds"],
        interval["reason"],
    )


def _remote_intervals(
    base_url, cameras, archive_dir, login, password, skipped_intervals=None,
):
    if skipped_intervals is None:
        skipped_intervals = []
    names = {}
    urls = {}
    for camera in cameras:
        urls[camera] = f"{base_url.rstrip('/')}/{quote(camera, safe='')}/"
        names[camera] = _remote_mkv_names(urls[camera], login, password)
        if not names[camera]:
            raise ValueError(f"remote camera archive is empty: {camera}")
        (Path(archive_dir) / camera).mkdir()
        log.info(
            "remote camera discovered camera=%s files=%d",
            camera, len(names[camera]),
        )

    active = {}
    indexes = {camera: 0 for camera in cameras}

    def advance(camera):
        index = indexes[camera]
        if index >= len(names[camera]):
            return False
        name = names[camera][index]
        destination = Path(archive_dir) / camera / name
        log.info("download started camera=%s file=%s", camera, name)
        _download_mkv(
            urljoin(urls[camera], quote(name, safe="")),
            destination,
            login,
            password,
        )
        fps, frame_count, start_pts = _video_info(destination)
        start = datetime.strptime(destination.stem, "%Y%m%d-%H%M%S") \
            + timedelta(seconds=start_pts)
        end = start + timedelta(seconds=frame_count / fps)
        previous = active.get(camera)
        active[camera] = (destination, start, end)
        indexes[camera] += 1
        if previous is not None:
            previous[0].unlink(missing_ok=True)
        log.info(
            "download completed camera=%s file=%s bytes=%d interval=%s..%s",
            camera, name, destination.stat().st_size, start, end,
        )
        return True

    with ThreadPoolExecutor(max_workers=len(cameras)) as downloads:
        initial = dict(zip(cameras, downloads.map(advance, cameras)))
        if not all(initial.values()):
            return

        archive_start = min(item[1] for item in active.values())
        latest_end = max(item[2] for item in active.values())
        previous_end = None
        while True:
            start = max(item[1] for item in active.values())
            end = min(item[2] for item in active.values())
            if start < end:
                _record_skipped_interval(
                    skipped_intervals,
                    previous_end if previous_end is not None else archive_start,
                    start,
                )
                log.info("analysis interval ready start=%s end=%s", start, end)
                yield start, end
                previous_end = end
                expired = [
                    camera for camera, item in active.items() if item[2] == end
                ]
            else:
                expired = [
                    camera for camera, item in active.items() if item[2] <= start
                ]
            advanced = dict(zip(expired, downloads.map(advance, expired)))
            for camera, available in advanced.items():
                if available:
                    latest_end = max(latest_end, active[camera][2])
            if not all(advanced.values()):
                _record_skipped_interval(
                    skipped_intervals,
                    previous_end if previous_end is not None else archive_start,
                    latest_end,
                )
                return


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
        "-show_entries", "stream=avg_frame_rate,nb_read_packets,start_time",
        "-of", "json", str(path),
    ], text=True)
    stream = json.loads(output)["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    frame_count = int(stream["nb_read_packets"])
    start_pts = float(stream.get("start_time", 0))
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"invalid video metadata for {path}: fps={fps}, frames={frame_count}"
        )
    return fps, frame_count, start_pts


def common_archive_interval(archive_dir, cameras):
    """Return the longest exact interval covered by every selected camera."""
    common = None
    for camera in cameras:
        camera_dir = Path(archive_dir) / camera
        files = _mkv_files(camera_dir)
        if not files:
            raise ValueError(f"camera archive is empty: {camera_dir}")
        ranges = []
        for item in files:
            fps, frame_count, start_pts = _video_info(
                camera_dir / item["name"]
            )
            start = item["timestamp"] + timedelta(seconds=start_pts)
            ranges.append((
                start,
                start + timedelta(seconds=frame_count / fps),
            ))
        merged = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        common = merged if common is None else [
            (max(left_start, right_start), min(left_end, right_end))
            for left_start, left_end in common
            for right_start, right_end in merged
            if max(left_start, right_start) < min(left_end, right_end)
        ]

    if not common:
        raise ValueError("selected camera archives do not overlap")
    return max(common, key=lambda interval: interval[1] - interval[0])


def build_archive_plan(
    archive_dir, start_time, end_time, cameras, interval_ms=None,
):
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
    for camera in cameras:
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
            fps, frame_count, start_pts = info[item["name"]]
            offset_seconds = (
                (target - item["timestamp"]).total_seconds() - start_pts
            )
            frame_index = min(
                max(round(offset_seconds * fps), 0),
                frame_count - 1,
            )
            pts_seconds = start_pts + frame_index / fps
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
            command = [
                "ffmpeg", "-v", "error", "-threads", "1",
                "-skip_loop_filter", "all",
                "-i", str(Path(archive_dir) / camera / mkv),
                "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
                "-fps_mode", "passthrough",
                "-frames:v", str(max(needed) + 1),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            remaining = set(needed)
            for index in range(max(needed) + 1):
                chunk = _read_exactly(process.stdout, frame_bytes)
                if chunk is None:
                    break
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
