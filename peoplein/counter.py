"""Detect people, count door crossings, and write run diagnostics."""

import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .database import record_passage

DIAGNOSTIC_CONFIDENCE = 0.01


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


@dataclass
class Track:
    foot: tuple
    side: int
    side_point: tuple
    misses: int = 0


class DoorCounter:
    def __init__(
        self, cameras, model_path, confidence, agreement_seconds,
        crossing_margin_px, database_path, app_version, detector=None,
        diagnostics_path=None, evidence_dir=None,
    ):
        self.cameras = cameras
        self.confidence = confidence
        self.detector = detector or PersonDetector(
            model_path, DIAGNOSTIC_CONFIDENCE,
        )
        self.agreement_seconds = agreement_seconds
        self.crossing_margin_px = crossing_margin_px
        self.database_path = database_path
        self.app_version = app_version
        self.tracks = {camera: {} for camera in cameras}
        self.next_track_id = 1
        self.events = []
        self.observations = []
        self.observations_by_camera = Counter()
        self.mismatch_pairs = set()
        self.frame_history = {
            camera: deque(maxlen=7) for camera in cameras
        }
        self.pending_evidence = []
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=False)
        self.diagnostics = None
        if diagnostics_path:
            path = Path(diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.diagnostics = path.open("x", encoding="utf-8")

    def update(self, camera, frame, timestamp):
        self._expire_events(timestamp)
        geometry = self.cameras[camera]
        line = geometry["line"]
        started = time.monotonic()
        raw_detections = [
            (bbox, score) for bbox, score in self.detector(frame)
            if score >= DIAGNOSTIC_CONFIDENCE
        ]
        detections, diagnostic_only = [], []
        for bbox, score in raw_detections:
            target = detections if score >= self.confidence or (
                score >= geometry.get("door_confidence", self.confidence)
                and self._distance_to_segment(
                    (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3]), line,
                ) <= geometry.get("door_confidence_radius_px", 0)
            ) else diagnostic_only
            target.append((bbox, score))
        inference_ms = round((time.monotonic() - started) * 1000, 3)
        feet = [
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
                (math.dist(track.foot, foot), track_id, index)
                for track_id, track in tracks.items()
                for index, foot in enumerate(feet)
            ),
            key=lambda item: item[0],
        )
        used_tracks, used_feet = set(), set()
        labels, crossings = [], []
        for distance, track_id, index in matches:
            if distance > 120 or track_id in used_tracks or index in used_feet:
                continue
            used_tracks.add(track_id)
            used_feet.add(index)
            direction, previous_side = self._move_track(
                tracks[track_id], feet[index], line,
            )
            bbox, score = detections[index]
            labels.append((track_id, bbox, score, tracks[track_id].side))
            self._track_diagnostic(
                timestamp, camera, track_id, bbox, score, feet[index],
                tracks[track_id].side, round(distance, 3), "matched", line,
            )
            if direction is not None:
                crossings.append((
                    track_id, geometry["directions"][direction], feet[index],
                    previous_side, tracks[track_id].side,
                ))

        for index, foot in enumerate(feet):
            if index in used_feet:
                continue
            side = self._side(foot, line)
            track_id = self.next_track_id
            tracks[track_id] = Track(foot, side, foot)
            self.next_track_id += 1
            bbox, score = detections[index]
            labels.append((track_id, bbox, score, side))
            self._track_diagnostic(
                timestamp, camera, track_id, bbox, score, foot, side,
                None, "new", line,
            )
        labels.extend(
            (None, bbox, score, None) for bbox, score in diagnostic_only
        )
        self.frame_history[camera].append((frame.copy(), timestamp, labels))
        for track_id in [
            track_id for track_id, track in tracks.items() if track.misses > 5
        ]:
            del tracks[track_id]
        for track_id, direction, foot, previous_side, side in crossings:
            self._passage(
                timestamp, camera, direction, track_id, foot,
                previous_side, side,
            )
        self._save_ready_evidence()

    def _track_diagnostic(
        self, timestamp, camera, track_id, bbox, score, foot, side,
        match_distance, state, line,
    ):
        self._diagnostic(
            "track_update", timestamp, camera,
            track_id=track_id,
            state=state,
            bbox=list(map(int, bbox)),
            detection_confidence=round(float(score), 6),
            foot=[round(value, 3) for value in foot],
            side=self._side_name(side),
            signed_distance_to_line=round(self._signed_distance(foot, line), 3),
            match_distance=match_distance,
        )

    def _move_track(self, track, foot, line):
        track.misses = 0
        track.foot = foot
        side = self._side(foot, line)
        if side == 0:
            return None, None
        if track.side == 0 or side == track.side:
            track.side = side
            track.side_point = foot
            return None, None
        previous_side = track.side
        crossed = self._segments_intersect(track.side_point, foot, *line)
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

    def _side(self, point, line):
        distance = self._signed_distance(point, line)
        if distance < -self.crossing_margin_px:
            return -1
        if distance > self.crossing_margin_px:
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
        self.observations.append(observation)
        self.observations_by_camera[camera] += 1
        self._diagnostic(
            "line_crossing", timestamp, camera,
            observation_id=observation_id,
            track_id=track_id,
            direction=direction,
            foot=[round(value, 3) for value in foot],
            previous_side=self._side_name(previous_side),
            side=self._side_name(side),
        )

        # ponytail: time-only agreement; add cross-camera ReID if simultaneous
        # passages prove that this pairs different people.
        matches = [
            event for event in self.events
            if len(event["observations"]) == 1
            and event["direction"] == direction
            and camera != event["observations"][0]["camera"]
            and abs((timestamp - event["timestamp"]).total_seconds())
            <= self.agreement_seconds
        ]
        opposites = [
            event for event in self.events
            if len(event["observations"]) == 1
            and event["direction"] != direction
            and camera != event["observations"][0]["camera"]
            and abs((timestamp - event["timestamp"]).total_seconds())
            <= self.agreement_seconds
        ]
        if opposites:
            other = min(
                opposites,
                key=lambda item: abs(
                    (timestamp - item["timestamp"]).total_seconds()
                ),
            )["observations"][0]
            pair = frozenset((observation_id, other["id"]))
            if pair not in self.mismatch_pairs:
                self.mismatch_pairs.add(pair)
                self._diagnostic(
                    "direction_mismatch", timestamp,
                    observation_ids=sorted(pair),
                    cameras=[other["camera"], camera],
                    directions=[other["direction"], direction],
                )
        if matches:
            event = min(
                matches,
                key=lambda item: abs(
                    (timestamp - item["timestamp"]).total_seconds()
                ),
            )
            event["observations"].append(observation)
            delta = abs((timestamp - event["timestamp"]).total_seconds())
            self._diagnostic(
                "passage_agreement", timestamp,
                observation_ids=[item["id"] for item in event["observations"]],
                cameras=[item["camera"] for item in event["observations"]],
                direction=direction,
                delta_seconds=round(delta, 3),
            )
        else:
            self.events.append({
                "timestamp": timestamp,
                "direction": direction,
                "observations": [observation],
                "unconfirmed_logged": False,
            })
        if self.evidence_dir:
            self.pending_evidence.append((observation_id, timestamp))

    def _expire_events(self, timestamp, force=False):
        for event in self.events:
            if len(event["observations"]) != 1 or event["unconfirmed_logged"]:
                continue
            age = (timestamp - event["timestamp"]).total_seconds()
            if not force and age <= self.agreement_seconds:
                continue
            observation = event["observations"][0]
            opposite = any(
                pair for pair in self.mismatch_pairs
                if observation["id"] in pair
            )
            self._diagnostic(
                "passage_unconfirmed", timestamp, observation["camera"],
                observation_id=observation["id"],
                passage_time=self._timestamp(observation["timestamp"]),
                direction=observation["direction"],
                waited_seconds=round(max(age, 0), 3),
                reason=(
                    "direction_mismatch" if opposite
                    else "no_matching_crossing"
                ),
            )
            event["unconfirmed_logged"] = True

    def _save_ready_evidence(self):
        pending = []
        for observation_id, timestamp in self.pending_evidence:
            windows = {
                camera: self._evidence_window(camera, timestamp)
                for camera in self.cameras
            }
            if any(window is None for window in windows.values()):
                pending.append((observation_id, timestamp))
                continue
            for camera, window in windows.items():
                for offset, snapshot in zip(range(-3, 4), window):
                    self._save_evidence_frame(
                        observation_id, timestamp, camera, offset, snapshot,
                    )
        self.pending_evidence = pending

    def _evidence_window(self, camera, timestamp):
        history = list(self.frame_history[camera])
        if len(history) < 7:
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
        self, observation_id, timestamp, camera, offset, snapshot,
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
        for track_id, bbox, score, side in labels:
            x, y, width, height = map(int, bbox)
            color = (
                (0, 255, 0) if score >= self.confidence else (0, 165, 255)
            )
            cv2.rectangle(
                annotated, (x, y), (x + width, y + height), color, 2,
            )
            if track_id is not None:
                cv2.circle(
                    annotated, (x + width // 2, y + height),
                    4, (255, 0, 0), -1,
                )
            cv2.putText(
                annotated,
                (
                    f"track={track_id} conf={score:.2f} {self._side_name(side)}"
                    if track_id is not None else f"conf={score:.2f} diagnostic"
                ),
                (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            f"passage={observation_id} frame={offset:+d} "
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
        target = self.evidence_dir / (
            f"passage_{observation_id}_{camera}_frame_{offset:+d}.jpg"
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
        entered = sum(event["direction"] == "entry" for event in self.events)
        exited = sum(event["direction"] == "exit" for event in self.events)
        confirmed = sum(
            len(event["observations"]) > 1 for event in self.events
        )
        confidence = confirmed / len(self.events) if self.events else 1.0
        result = {
            "entered_total": entered,
            "exited_total": exited,
            "people_inside": max(entered - exited, 0),
            "people_inside_confidence": round(confidence, 6),
        }
        if timestamp is not None:
            self._diagnostic("counter_snapshot", timestamp, **result)
        return result

    def diagnostic_summary(self):
        return {
            "observations_by_camera": {
                camera: self.observations_by_camera[camera]
                for camera in self.cameras
            },
            "confirmed_passages": sum(
                len(event["observations"]) > 1 for event in self.events
            ),
            "unconfirmed_passages": sum(
                len(event["observations"]) == 1 for event in self.events
            ),
            "direction_mismatches": len(self.mismatch_pairs),
        }

    def finish(self, timestamp):
        self._expire_events(timestamp, force=True)

    def close(self):
        if self.diagnostics:
            self.diagnostics.close()
            self.diagnostics = None
