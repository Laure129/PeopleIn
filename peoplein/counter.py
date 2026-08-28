"""Detect people, count door crossings, and write run diagnostics."""

import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .database import record_passage

DIAGNOSTIC_CONFIDENCE = 0.01
MOTION_ACTIVITY_GAP_SECONDS = 3
PERSON_ANALYSIS_WINDOW_SECONDS = 5
MOTION_PROFILE_MIN_BINS = 2
MOTION_PROFILE_SCAN_SECONDS = 6


class PersonDetector:
    def __init__(self, model_path, confidence):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.confidence = confidence
        self.input_size = 320

    def __call__(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        left = (self.input_size - resized_width) // 2
        top = (self.input_size - resized_height) // 2
        image = np.full(
            (self.input_size, self.input_size, 3), 114, dtype=np.uint8,
        )
        image[
            top:top + resized_height, left:left + resized_width,
        ] = cv2.resize(frame, (resized_width, resized_height))
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255, (self.input_size, self.input_size), swapRB=True,
        )
        self.net.setInput(blob)
        detections = np.squeeze(self.net.forward())
        if detections.shape[0] < detections.shape[1]:
            detections = detections.T

        boxes, scores = [], []
        for cx, cy, box_width, box_height, score in detections[:, :5]:
            if score < self.confidence:
                continue
            x1 = max(0, round((cx - box_width / 2 - left) / scale))
            y1 = max(0, round((cy - box_height / 2 - top) / scale))
            x2 = min(width, round((cx + box_width / 2 - left) / scale))
            y2 = min(height, round((cy + box_height / 2 - top) / scale))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
                scores.append(float(score))
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence, 0.45,
        )
        return [
            (boxes[int(index)], scores[int(index)])
            for index in np.asarray(indices).reshape(-1)
        ]


class OpenVinoPersonDetector:
    def __init__(self, model_path, confidence):
        from openvino import Core

        self.model = Core().compile_model(str(model_path), "CPU")
        self.input = self.model.input(0)
        self.output = self.model.output(0)
        _, _, height, width = self.input.shape
        self.height, self.width = int(height), int(width)
        self.confidence = confidence

    def __call__(self, frame):
        height, width = frame.shape[:2]
        image = cv2.resize(frame, (self.width, self.height))
        image = image.transpose(2, 0, 1)[None]
        detections = self.model([image])[self.output].reshape(-1, 7)
        result = []
        for image_id, label, score, x1, y1, x2, y2 in detections:
            if image_id < 0:
                break
            if label != 0 or score < self.confidence:
                continue
            box = (
                max(0, round(x1 * width)),
                max(0, round(y1 * height)),
                min(width, round(x2 * width)),
                min(height, round(y2 * height)),
            )
            if box[2] > box[0] and box[3] > box[1]:
                result.append((
                    (box[0], box[1], box[2] - box[0], box[3] - box[1]),
                    float(score),
                ))
        return result


def person_detector(model_path, confidence):
    if Path(model_path).suffix == ".xml":
        return OpenVinoPersonDetector(model_path, confidence)
    return PersonDetector(model_path, confidence)


@dataclass
class Track:
    foot: tuple
    side: int
    side_point: tuple
    misses: int = 0
    crossed: bool = False


class DoorCounter:
    def __init__(
        self, cameras, model_path, confidence, agreement_seconds,
        crossing_margin_px, database_path, app_version, detector=None,
        diagnostics_path=None, evidence_dir=None,
        door_motion_activity_dir=None,
        motion_profile_bin_frames=5,
        reference_events=(),
    ):
        self.cameras = cameras
        self.confidence = confidence
        self.detector = detector or person_detector(
            model_path, DIAGNOSTIC_CONFIDENCE,
        )
        self.motion = {
            camera: {
                "previous_gray": None,
                "activity_gray": None,
                "activity_samples": 0,
                "mask": None,
            }
            for camera, geometry in cameras.items()
            if "motion_roi" in geometry
        }
        if motion_profile_bin_frames <= 0:
            raise ValueError("motion_profile_bin_frames must be positive")
        self.motion_profile_bin_frames = motion_profile_bin_frames
        self.motion_profiles = {
            camera: self._new_motion_profile()
            for camera in self.motion
        }
        self.door_profile_entries = []
        self.agreement_seconds = agreement_seconds
        self.crossing_margin_px = crossing_margin_px
        self.database_path = database_path
        self.app_version = app_version
        self.tracks = {camera: {} for camera in cameras}
        self.next_track_id = 1
        self.events = []
        self.door_motion_activity = []
        self.full_camera_motion_activity = []
        self.door_motion_activity_start = 0
        self.full_camera_motion_activity_start = 0
        self.pending_frames = {
            camera: deque() for camera in cameras
        }
        self.frame_history = {
            camera: deque(maxlen=7) for camera in cameras
        }
        self.pending_evidence = [
            (
                "reference", event["id"], event["timestamp"],
                tuple(self.cameras), event["direction"], None,
            )
            for event in reference_events
        ]
        self.matched_reference_events = set()
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=False)
        self.door_motion_activity_dir = (
            Path(door_motion_activity_dir)
            if door_motion_activity_dir else None
        )
        if self.door_motion_activity_dir:
            self.door_motion_activity_dir.mkdir(parents=True, exist_ok=False)
        self.diagnostics = None
        if diagnostics_path:
            path = Path(diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.diagnostics = path.open("x", encoding="utf-8")

    def update(self, camera, frame, timestamp):
        geometry = self.cameras[camera]
        if camera in self.motion:
            started = time.monotonic()
            (
                door_motion_points, flow_vectors,
                full_camera_motion_points, profile_points,
            ) = self._motion_flow(camera, frame)
            inference_ms = round((time.monotonic() - started) * 1000, 3)
            self._diagnostic(
                "motion_flow", timestamp, camera,
                inference_ms=inference_ms,
                door_motion_points=door_motion_points,
                full_camera_motion_points=full_camera_motion_points,
                motion_profile_points=profile_points,
            )
            self._update_motion_profile(camera, timestamp, profile_points)
            self.pending_frames[camera].append((
                frame.copy(), timestamp, flow_vectors,
            ))
            if door_motion_points >= geometry["motion_min_points"]:
                self.door_motion_activity.append({
                    "timestamp": timestamp,
                    "camera": camera,
                    "motion_points": door_motion_points,
                    "activity": "door_motion_activity",
                })
                if self.door_motion_activity_dir:
                    self._save_evidence_frame(
                        "door_motion_activity",
                        f"{len(self.door_motion_activity)}_points_"
                        f"{door_motion_points}",
                        timestamp, camera, 0,
                        (frame, timestamp, flow_vectors),
                        self.door_motion_activity_dir,
                    )
            if full_camera_motion_points >= geometry["motion_min_points"]:
                self.full_camera_motion_activity.append({
                    "timestamp": timestamp,
                    "camera": camera,
                    "motion_points": full_camera_motion_points,
                    "activity": "full_camera_motion_activity",
                })
            if (
                door_motion_points >= geometry["motion_min_points"]
                or full_camera_motion_points >= geometry["motion_min_points"]
            ):
                self._match_motion(timestamp)
            self._flush_frames(timestamp)
            return

        if self.motion:
            self.pending_frames[camera].append((frame.copy(), timestamp, []))
            self._flush_frames(timestamp)
            return

        self._expire_events(timestamp)
        self._analyze_people(camera, frame, timestamp)

    def _flush_frames(self, current_time, force=False):
        cutoff = current_time - timedelta(
            seconds=PERSON_ANALYSIS_WINDOW_SECONDS
        )
        ready = []
        for order, (camera, frames) in enumerate(self.pending_frames.items()):
            while frames and (force or frames[0][1] < cutoff):
                frame, timestamp, labels = frames.popleft()
                ready.append((timestamp, order, camera, frame, labels))

        for timestamp, _, camera, frame, labels in sorted(
            ready, key=lambda item: item[:2],
        ):
            self._expire_events(timestamp)
            if camera in self.motion:
                self.frame_history[camera].append((frame, timestamp, labels))
                self._save_ready_evidence(timestamp)
            elif any(
                abs((timestamp - activity["timestamp"]).total_seconds())
                <= PERSON_ANALYSIS_WINDOW_SECONDS
                for activity in self.full_camera_motion_activity[
                    self.full_camera_motion_activity_start:
                ]
            ):
                self._analyze_people(camera, frame, timestamp)
            else:
                self.tracks[camera].clear()
                self.frame_history[camera].append((frame, timestamp, []))
                self._save_ready_evidence(timestamp)

    def _analyze_people(self, camera, frame, timestamp):
        geometry = self.cameras[camera]
        line = geometry["line"]
        started = time.monotonic()
        raw_detections = [
            (bbox, score) for bbox, score in self.detector(frame)
            if score >= DIAGNOSTIC_CONFIDENCE
        ]
        detections = []
        for bbox, score in raw_detections:
            if score >= self.confidence or (
                score >= geometry.get("door_confidence", self.confidence)
                and self._distance_to_segment(
                    (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3]), line,
                ) <= geometry.get("door_confidence_radius_px", 0)
            ):
                detections.append((bbox, score))
        inference_ms = round((time.monotonic() - started) * 1000, 3)
        points = [
            (x + width / 2, y + height)
            for (x, y, width, height), _ in detections
        ]
        self._diagnostic(
            "detection", timestamp, camera,
            inference_ms=inference_ms,
            detection_count=len(raw_detections),
            detections=[
                {
                    "bbox": list(map(int, bbox)),
                    "confidence": round(float(score), 6),
                }
                for bbox, score in raw_detections
            ],
        )

        tracks = self.tracks[camera]
        for track in tracks.values():
            track.misses += 1

        # ponytail: greedy local matching; add ReID only after measured ID swaps.
        matches = sorted(
            (
                (math.dist(track.foot, point), track_id, index)
                for track_id, track in tracks.items()
                for index, point in enumerate(points)
            ),
            key=lambda item: item[0],
        )
        used_tracks, used_points = set(), set()
        labels, crossings = [], []
        for distance, track_id, index in matches:
            if distance > 120 or track_id in used_tracks or index in used_points:
                continue
            used_tracks.add(track_id)
            used_points.add(index)
            direction, previous_side = self._move_track(
                tracks[track_id], points[index], line,
            )
            bbox, score = detections[index]
            labels.append((
                track_id, bbox, score, tracks[track_id].side, points[index],
            ))
            self._track_diagnostic(
                timestamp, camera, track_id, bbox, score, points[index],
                tracks[track_id].side, round(distance, 3), "matched", line,
            )
            if direction is not None:
                tracks[track_id].crossed = True
                crossings.append((
                    track_id, geometry["directions"][direction], points[index],
                    previous_side, tracks[track_id].side,
                ))

        for index, point in enumerate(points):
            if index in used_points:
                continue
            side = self._side(point, line)
            track_id = self.next_track_id
            tracks[track_id] = Track(point, side, point)
            self.next_track_id += 1
            bbox, score = detections[index]
            labels.append((track_id, bbox, score, side, point))
            self._track_diagnostic(
                timestamp, camera, track_id, bbox, score, point, side,
                None, "new", line,
            )
        self.frame_history[camera].append((frame.copy(), timestamp, labels))
        for track_id in [
            track_id for track_id, track in tracks.items()
            if track.misses > 5
        ]:
            del tracks[track_id]
        for track_id, direction, foot, previous_side, side in crossings:
            self._passage(
                timestamp, camera, direction, track_id, foot,
                previous_side, side,
            )
        self._save_ready_evidence(timestamp)

    def _motion_flow(self, camera, frame):
        geometry = self.cameras[camera]
        state = self.motion[camera]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous = state["previous_gray"]
        state["previous_gray"] = gray
        if previous is None:
            state["activity_gray"] = gray
            return 0, [], 0, {"entry": 0, "exit": 0}
        if state["mask"] is None:
            roi = np.zeros(gray.shape, dtype=np.uint8)
            band = np.zeros_like(roi)
            cv2.rectangle(roi, *geometry["motion_roi"], 255, -1)
            cv2.line(
                band, *geometry["line"], 255,
                geometry["motion_band_width_px"] * 2,
            )
            state["mask"] = cv2.bitwise_and(roi, band)
        profile_minimum = geometry.get(
            "motion_profile_min_displacement_px",
            geometry["motion_min_displacement_px"],
        )
        vectors = self._flow_vectors(
            previous, gray, state["mask"], profile_minimum,
        )
        profile_points = Counter()
        for start, end in vectors:
            if not self._motion_direction_allowed(start, end, geometry):
                continue
            direction = (
                "left_to_right"
                if self._signed_distance(end, geometry["line"])
                > self._signed_distance(start, geometry["line"])
                else "right_to_left"
            )
            profile_points[geometry["directions"][direction]] += 1
        state["activity_samples"] += 1
        if state["activity_samples"] < self.motion_profile_bin_frames:
            return 0, [], 0, {
                direction: profile_points[direction]
                for direction in ("entry", "exit")
            }
        activity_previous = state["activity_gray"]
        state["activity_gray"] = gray
        state["activity_samples"] = 0
        door_vectors = [
            (start, end) for start, end in self._flow_vectors(
                activity_previous, gray, state["mask"],
                geometry["motion_min_displacement_px"],
            )
            if self._motion_direction_allowed(start, end, geometry)
        ]
        full_camera_vectors = self._flow_vectors(
            activity_previous, gray, None,
            geometry["motion_min_displacement_px"],
        )
        return (
            len(door_vectors), door_vectors, len(full_camera_vectors),
            {direction: profile_points[direction] for direction in (
                "entry", "exit",
            )},
        )

    @staticmethod
    def _new_motion_profile():
        return {
            "bin_start": None,
            "samples": 0,
            "entry_points": 0,
            "exit_points": 0,
            "opened_until": None,
            "active_bins": [],
        }

    def _update_motion_profile(self, camera, timestamp, points):
        state = self.motion_profiles[camera]
        if state["bin_start"] is None:
            state["bin_start"] = timestamp
        state["samples"] += 1
        state["entry_points"] += points["entry"]
        state["exit_points"] += points["exit"]
        if state["samples"] < self.motion_profile_bin_frames:
            return
        self._motion_profile_bin(
            camera, state["bin_start"],
            state["entry_points"], state["exit_points"],
        )
        state["bin_start"] = None
        state["samples"] = 0
        state["entry_points"] = 0
        state["exit_points"] = 0

    def _motion_profile_bin(
        self, camera, timestamp, entry_points, exit_points,
    ):
        state = self.motion_profiles[camera]
        geometry = self.cameras[camera]
        if exit_points >= geometry.get(
            "motion_profile_open_min_points", geometry["motion_min_points"],
        ):
            self._finish_motion_profile(camera)
            state["opened_until"] = timestamp + timedelta(
                seconds=MOTION_PROFILE_SCAN_SECONDS,
            )
        if state["opened_until"] is None:
            return
        if timestamp > state["opened_until"]:
            self._finish_motion_profile(camera)
            state["opened_until"] = None
            return
        if entry_points >= geometry.get(
            "motion_profile_min_points", geometry["motion_min_points"],
        ):
            state["active_bins"].append((timestamp, entry_points))
        else:
            self._finish_motion_profile(camera)

    def _finish_motion_profile(self, camera):
        state = self.motion_profiles[camera]
        bins = state["active_bins"]
        if len(bins) >= MOTION_PROFILE_MIN_BINS:
            timestamp, peak = max(bins, key=lambda item: item[1])
            entry = {
                "timestamp": timestamp,
                "camera": camera,
                "start": bins[0][0],
                "end": bins[-1][0],
                "bins": len(bins),
                "peak_motion_points": peak,
            }
            self.door_profile_entries.append(entry)
            self._diagnostic(
                "door_profile_entry", timestamp, camera,
                start=self._timestamp(entry["start"]),
                end=self._timestamp(entry["end"]),
                bins=entry["bins"],
                peak_motion_points=peak,
            )
        state["active_bins"] = []

    @staticmethod
    def _flow_vectors(previous, gray, mask, minimum_displacement):
        points = cv2.goodFeaturesToTrack(
            previous, mask=mask, maxCorners=300,
            qualityLevel=0.01, minDistance=4, blockSize=5,
        )
        if points is None:
            return []
        current, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(21, 21), maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03,
            ),
        )
        if current is None or status is None:
            return []
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous, current, None, winSize=(21, 21), maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03,
            ),
        )
        if backward is None or backward_status is None:
            return []
        vectors = []
        for start, end, returned, ok, backward_ok in zip(
            points.reshape(-1, 2), current.reshape(-1, 2),
            backward.reshape(-1, 2), status.ravel(), backward_status.ravel(),
        ):
            if not ok or not backward_ok or math.dist(start, returned) > 1.5:
                continue
            if math.dist(start, end) < minimum_displacement:
                continue
            vectors.append((start, end))
        return vectors

    @staticmethod
    def _motion_direction_allowed(start, end, geometry):
        if not geometry.get("motion_perpendicular_only", False):
            return True
        (x1, y1), (x2, y2) = geometry["line"]
        motion_x, motion_y = end[0] - start[0], end[1] - start[1]
        line_x, line_y = x2 - x1, y2 - y1
        return abs(motion_x * line_x + motion_y * line_y) <= abs(
            motion_x * line_y - motion_y * line_x
        )

    def _track_diagnostic(
        self, timestamp, camera, track_id, bbox, score, foot, side,
        match_distance, state, line,
    ):
        self._diagnostic(
            "track_update",
            timestamp, camera,
            track_id=track_id,
            state=state,
            bbox=list(map(int, bbox)),
            detection_confidence=round(float(score), 6),
            **{
                "foot": [
                    round(value, 3) for value in foot
                ],
            },
            side=self._side_name(side),
            signed_distance_to_line=round(self._signed_distance(foot, line), 3),
            match_distance=match_distance,
        )

    def _move_track(self, track, foot, line, margin=None):
        track.misses = 0
        track.foot = foot
        side = self._side(foot, line, margin)
        if side == 0:
            return None, None
        if side == track.side:
            track.side = side
            track.side_point = foot
            return None, None
        previous_side = -side if track.side == 0 else track.side
        crossed = track.side == 0 or self._segments_intersect(
            track.side_point, foot, *line,
        )
        track.side = side
        track.side_point = foot
        if not crossed:
            return None, None
        return (
            "left_to_right" if (previous_side, side) == (-1, 1)
            else "right_to_left",
            previous_side,
        )

    def _signed_distance(self, point, line):
        (x1, y1), (x2, y2) = line
        return (
            (point[0] - x1) * (y2 - y1)
            - (point[1] - y1) * (x2 - x1)
        ) / math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _distance_to_segment(point, line):
        start, end = line
        dx, dy = end[0] - start[0], end[1] - start[1]
        fraction = max(0, min(1, (
            (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
        ) / (dx * dx + dy * dy)))
        closest = (start[0] + fraction * dx, start[1] + fraction * dy)
        return math.dist(point, closest)

    def _side(self, point, line, margin=None):
        distance = self._signed_distance(point, line)
        margin = self.crossing_margin_px if margin is None else margin
        if distance < -margin:
            return -1
        if distance > margin:
            return 1
        return 0

    @staticmethod
    def _side_name(side):
        return {-1: "left", 0: "line", 1: "right"}[side]

    @staticmethod
    def _segments_intersect(start, end, line_start, line_end):
        move_x, move_y = end[0] - start[0], end[1] - start[1]
        line_x = line_end[0] - line_start[0]
        line_y = line_end[1] - line_start[1]
        denominator = move_x * line_y - move_y * line_x
        if denominator == 0:
            return False
        offset_x = line_start[0] - start[0]
        offset_y = line_start[1] - start[1]
        move_fraction = (offset_x * line_y - offset_y * line_x) / denominator
        line_fraction = (offset_x * move_y - offset_y * move_x) / denominator
        return 0 <= move_fraction <= 1 and 0 <= line_fraction <= 1

    def _passage(
        self, timestamp, camera, direction, track_id, foot,
        previous_side, side,
    ):
        observation_id = record_passage(
            self.database_path, timestamp, self.app_version, camera, direction,
        )
        observation = {
            "id": observation_id,
            "timestamp": timestamp,
            "camera": camera,
            "direction": direction,
        }
        self._diagnostic(
            "line_crossing", timestamp, camera,
            observation_id=observation_id,
            track_id=track_id,
            direction=direction,
            foot=[round(value, 3) for value in foot],
            previous_side=self._side_name(previous_side),
            side=self._side_name(side),
        )
        self.events.append({
            "timestamp": timestamp,
            "direction": direction,
            "observations": [observation],
            "confirmed": False,
            "unconfirmed_logged": False,
        })
        self._match_motion(timestamp)
        if self.evidence_dir:
            self.pending_evidence.append((
                "passage", observation_id, timestamp, tuple(self.cameras),
                direction, None,
            ))

    def _match_motion(self, timestamp, force=False):
        for event in self.events:
            if event["confirmed"]:
                continue
            age = (timestamp - event["timestamp"]).total_seconds()
            if not force and age <= self.agreement_seconds:
                continue
            door_candidates = [
                candidate for candidate in self.door_motion_activity[
                    self.door_motion_activity_start:
                ]
                if abs((event["timestamp"] - candidate["timestamp"])
                       .total_seconds()) <= self.agreement_seconds
            ]
            candidates = door_candidates or [
                candidate for candidate in self.full_camera_motion_activity[
                    self.full_camera_motion_activity_start:
                ]
                if abs((event["timestamp"] - candidate["timestamp"])
                       .total_seconds()) <= self.agreement_seconds
            ]
            if not candidates:
                continue
            candidate = min(
                candidates,
                key=lambda item: abs(
                    (event["timestamp"] - item["timestamp"]).total_seconds()
                ),
            )
            delta = abs(
                (event["timestamp"] - candidate["timestamp"]).total_seconds()
            )
            event["confirmed"] = True
            event["confirmed_at"] = timestamp
            event["motion_candidate"] = candidate
            observation = event["observations"][0]
            self._diagnostic(
                "passage_agreement", timestamp,
                observation_ids=[observation["id"]],
                cameras=[observation["camera"], candidate["camera"]],
                direction=event["direction"],
                delta_seconds=round(delta, 3),
                motion_points=candidate["motion_points"],
                activity=candidate["activity"],
            )

    def _expire_events(self, timestamp, force=False):
        self._match_motion(timestamp, force)
        for event in self.events:
            if event["confirmed"] or event["unconfirmed_logged"]:
                continue
            age = (timestamp - event["timestamp"]).total_seconds()
            if not force and age <= self.agreement_seconds:
                continue
            observation = event["observations"][0]
            self._diagnostic(
                "passage_unconfirmed", timestamp, observation["camera"],
                observation_id=observation["id"],
                passage_time=self._timestamp(observation["timestamp"]),
                direction=observation["direction"],
                waited_seconds=round(max(age, 0), 3),
                reason="no_motion_activity",
            )
            event["unconfirmed_logged"] = True

    def _save_ready_evidence(self, current_time, force=False):
        pending = []
        for kind, event_id, timestamp, cameras, direction, windows \
                in self.pending_evidence:
            if windows is None:
                windows = {
                    camera: self._evidence_window(camera, timestamp)
                    for camera in cameras
                }
            if any(window is None for window in windows.values()):
                pending.append((
                    kind, event_id, timestamp, cameras, direction, None,
                ))
                continue
            if kind == "reference":
                if not force and (current_time - timestamp).total_seconds() < 1:
                    pending.append((
                        kind, event_id, timestamp, cameras, direction, windows,
                    ))
                    continue
                matches = [
                    event for event in self.events
                    if event["direction"] == direction
                    and event["observations"][0]["id"]
                    not in self.matched_reference_events
                    and abs(
                        (event["timestamp"] - timestamp).total_seconds()
                    ) <= 1
                ]
                if matches:
                    match = min(
                        matches,
                        key=lambda event: abs(
                            (event["timestamp"] - timestamp).total_seconds()
                        ),
                    )
                    self.matched_reference_events.add(
                        match["observations"][0]["id"]
                    )
                    continue
            for camera, window in windows.items():
                for offset, snapshot in zip(range(-3, 4), window):
                    self._save_evidence_frame(
                        kind, event_id, timestamp, camera, offset, snapshot,
                        direction=direction,
                    )
        self.pending_evidence = pending

    def _evidence_window(self, camera, timestamp):
        history = list(self.frame_history[camera])
        if len(history) < 7:
            return None
        if not history[0][1] <= timestamp <= history[-1][1]:
            return None
        center = min(
            range(len(history)),
            key=lambda index: abs(
                (history[index][1] - timestamp).total_seconds()
            ),
        )
        if center < 3 or len(history) - center <= 3:
            return None
        return history[center - 3:center + 4]

    def _save_evidence_frame(
        self, kind, event_id, timestamp, camera, offset, snapshot,
        directory=None, direction=None,
    ):
        frame, frame_time, labels = snapshot
        annotated = frame.copy()
        radius = round(
            self.cameras[camera].get("door_confidence_radius_px", 0)
        )
        if radius:
            overlay = annotated.copy()
            cv2.line(
                overlay, *self.cameras[camera]["line"],
                (0, 165, 255), radius * 2, cv2.LINE_AA,
            )
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
        cv2.line(
            annotated, *self.cameras[camera]["line"],
            (0, 0, 255), 2, cv2.LINE_AA,
        )
        if camera in self.motion:
            for start, end in labels:
                color = (255, 255, 0)
                start, end = tuple(map(round, start)), tuple(map(round, end))
                cv2.arrowedLine(
                    annotated, start, end, color, 2, cv2.LINE_AA,
                    tipLength=0.25,
                )
                cv2.circle(annotated, end, 3, color, -1, cv2.LINE_AA)
        else:
            for track_id, bbox, score, side, point in labels:
                if score < self.confidence:
                    continue
                x, y, width, height = map(int, bbox)
                color = (0, 255, 0)
                cv2.rectangle(
                    annotated, (x, y), (x + width, y + height), color, 2,
                )
                if track_id is not None:
                    cv2.circle(
                        annotated, tuple(map(round, point)),
                        4, (255, 0, 0), -1,
                    )
                cv2.putText(
                    annotated,
                    (
                        f"track={track_id} conf={score:.2f} "
                        f"{self._side_name(side)}"
                        if track_id is not None
                        else f"conf={score:.2f} diagnostic"
                    ),
                    (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA,
                )
        cv2.putText(
            annotated,
            f"{kind}={event_id} frame={offset:+d} "
            f"event={self._timestamp(timestamp)}",
            (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"camera={camera} frame={self._timestamp(frame_time)}",
            (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        target = (directory or self.evidence_dir) / (
            f"{kind}_{event_id}_{f'{direction}_' if direction else ''}"
            f"{camera}_frame_{offset:+d}.jpg"
        )
        if not cv2.imwrite(str(target), annotated):
            raise RuntimeError(f"cannot write evidence image: {target}")

    def _diagnostic(self, event, timestamp, camera=None, **values):
        if not self.diagnostics:
            return
        record = {
            "archive_time": self._timestamp(timestamp),
            "wall_time": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "app_version": self.app_version,
            "camera": camera,
            "event": event,
            **values,
        }
        self.diagnostics.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.diagnostics.flush()

    @staticmethod
    def _timestamp(value):
        return value.isoformat(sep=" ", timespec="milliseconds")

    def snapshot(self, timestamp=None):
        events = (
            self.events if timestamp is None else [
                event for event in self.events
                if event["timestamp"] < timestamp + timedelta(seconds=1)
            ]
        )
        entered = sum(event["direction"] == "entry" for event in events)
        exited = sum(event["direction"] == "exit" for event in events)
        confirmed = sum(
            event["confirmed"] and (
                timestamp is None
                or event.get("confirmed_at", event["timestamp"])
                < timestamp + timedelta(seconds=1)
            )
            for event in events
        )
        result = {
            "entered_total": entered,
            "exited_total": exited,
            "people_inside": max(entered - exited, 0),
            "door_profile_entered_total": sum(
                timestamp is None
                or entry["timestamp"] < timestamp + timedelta(seconds=1)
                for entry in self.door_profile_entries
            ),
            "passage_confirmation_ratio": (
                round(confirmed / len(events), 6)
                if events else None
            ),
        }
        if timestamp is not None:
            self._diagnostic("counter_snapshot", timestamp, **result)
        return result

    def diagnostic_summary(self):
        return {
            "door_motion_activity_intervals": self._activity_intervals(
                self.door_motion_activity,
            ),
            "full_camera_motion_activity_intervals": (
                self._activity_intervals(
                    self.full_camera_motion_activity,
                )
            ),
            "passages": [
                self._passage_summary(event) for event in self.events
            ],
        }

    def _activity_intervals(self, activities):
        result = []
        for camera in self.motion:
            previous = None
            for activity in (
                item for item in activities
                if item["camera"] == camera
            ):
                timestamp = activity["timestamp"]
                if (
                    previous is not None
                    and (timestamp - previous).total_seconds()
                    < MOTION_ACTIVITY_GAP_SECONDS
                ):
                    result[-1]["end"] = timestamp.isoformat(
                        timespec="milliseconds"
                    )
                else:
                    result.append({
                        "camera": camera,
                        "start": timestamp.isoformat(timespec="milliseconds"),
                        "end": timestamp.isoformat(timespec="milliseconds"),
                    })
                previous = timestamp
        return result

    def _passage_summary(self, event):
        observation = event["observations"][0]
        motion = event.get("motion_candidate")
        return {
            "timestamp": observation["timestamp"].isoformat(
                timespec="milliseconds"
            ),
            "direction": event["direction"],
            "camera": observation["camera"],
            "confirmation": (
                {
                    "activity": motion["activity"],
                    "camera": motion["camera"],
                    "timestamp": motion["timestamp"].isoformat(
                        timespec="milliseconds"
                    ),
                    "delta_seconds": round(abs(
                        (observation["timestamp"] - motion["timestamp"])
                        .total_seconds()
                    ), 3),
                } if motion else None
            ),
        }

    def finish(self, timestamp):
        if self.motion:
            self._flush_frames(timestamp, force=True)
            for camera in self.motion:
                self._finish_motion_profile(camera)
        self._expire_events(timestamp, force=True)
        self._save_ready_evidence(timestamp, force=True)

    def reset_stream(self):
        """Discard temporal state that cannot cross an archive gap."""
        for tracks in self.tracks.values():
            tracks.clear()
        for frames in self.pending_frames.values():
            frames.clear()
        for history in self.frame_history.values():
            history.clear()
        for state in self.motion.values():
            state["previous_gray"] = None
            state["activity_gray"] = None
            state["activity_samples"] = 0
        self.motion_profiles = {
            camera: self._new_motion_profile()
            for camera in self.motion
        }
        self.pending_evidence.clear()
        self.door_motion_activity_start = len(self.door_motion_activity)
        self.full_camera_motion_activity_start = len(
            self.full_camera_motion_activity
        )

    def close(self):
        if self.diagnostics:
            self.diagnostics.close()
            self.diagnostics = None
