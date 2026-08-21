"""Shared archive clock and camera synchronization checks."""

import time
from datetime import timedelta


class PlaybackClock:
    """One absolute playback clock shared by every camera."""

    def __init__(self, start_time, frame_interval_ms, speed):
        if frame_interval_ms <= 0:
            raise ValueError("frame_interval_ms must be greater than zero")
        if speed <= 0:
            raise ValueError("speed must be greater than zero")
        self.start_time = start_time
        self.archive_interval = frame_interval_ms / 1000.0
        self.wall_interval = self.archive_interval / speed
        self.started_at = time.monotonic()
        self.tick = 0

    @property
    def expected_archive_time(self):
        return self.start_time + timedelta(
            seconds=self.tick * self.archive_interval,
        )

    @property
    def playback_lag_ms(self):
        deadline = self.started_at + (self.tick + 1) * self.wall_interval
        return max(0, round((time.monotonic() - deadline) * 1000))

    def advance(self):
        self.tick += 1
        deadline = self.started_at + self.tick * self.wall_interval
        delay = deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def capture_needs_resync(expected_time, source_time, archive_interval):
    return (
        expected_time is not None
        and source_time is not None
        and abs((source_time - expected_time).total_seconds()) > archive_interval
    )


def archive_skew_ms(stats_by_camera):
    samples = [
        (stats.get("playback_tick"), stats.get("mkv_pts_timestamp"))
        for stats in stats_by_camera.values()
        if stats.get("playback_tick") is not None
        and stats.get("mkv_pts_timestamp") is not None
    ]
    if not samples:
        return None
    latest_tick = max(tick for tick, _ in samples)
    timestamps = [timestamp for tick, timestamp in samples if tick == latest_tick]
    if len(timestamps) < 2:
        return None
    return round((max(timestamps) - min(timestamps)) * 1000)
