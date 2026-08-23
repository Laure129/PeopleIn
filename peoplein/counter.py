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
MOTION_MARGIN_PX = 5


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
    ):
        self.cameras = cameras
        self.confidence = confidence
        self.detector = detector or person_detector(
            model_path, DIAGNOSTIC_CONFIDENCE,
        )
        self.motion = {
            camera: {"previous_gray": None, "mask": None}
            for camera, geometry in cameras.items()
            if "motion_roi" in geometry
        }
        self.agreement_seconds = agreement_seconds
        self.crossing_margin_px = crossing_margin_px
        self.database_path = database_path
        self.app_version = app_version
        self.tracks = {camera: {} for camera in cameras}
        self.next_track_id = 1
        self.events = []
        self.motion_activity = []
        self.observations = []
        self.observations_by_camera = Counter()
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
        if camera in self.motion:
            direction_points, axis_points, flow_vectors = self._motion_flow(
                camera, frame,
            )
            inference_ms = round((time.monotonic() - started) * 1000, 3)
            self._diagnostic(
                "motion_flow", timestamp, camera,
                inference_ms=inference_ms,
                point_count=sum(direction_points.values()),
                direction_points=dict(direction_points),
                axis_points=axis_points,
            )
            self.frame_history[camera].append((
                frame.copy(), timestamp, flow_vectors,
            ))
            if (
                sum(direction_points.values()) >= geometry["motion_min_points"]
                or axis_points >= geometry["motion_min_points"]
            ):
                self.motion_activity.append({
                    "timestamp": timestamp,
                    "camera": camera,
                    "direction_points": dict(direction_points),
                    "axis_points": axis_points,
                })
                self._match_motion(timestamp)
            self._save_ready_evidence()
            return

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
        self._save_ready_evidence()

    def _motion_flow(self, camera, frame):
        geometry = self.cameras[camera]
        state = self.motion[camera]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous = state["previous_gray"]
        state["previous_gray"] = gray
        if previous is None:
            return Counter(), 0, []
        if state["mask"] is None:
            roi = np.zeros(gray.shape, dtype=np.uint8)
            band = np.zeros_like(roi)
            cv2.rectangle(roi, *geometry["motion_roi"], 255, -1)
            cv2.line(
                band, *geometry["line"], 255,
                geometry["motion_band_width_px"] * 2,
            )
            state["mask"] = cv2.bitwise_and(roi, band)
        points = cv2.goodFeaturesToTrack(
            previous, mask=state["mask"], maxCorners=300,
            qualityLevel=0.01, minDistance=4, blockSize=5,
        )
        if points is None:
            return Counter(), 0, []
        current, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(21, 21), maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03,
            ),
        )
        if current is None or status is None:
            return Counter(), 0, []
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous, current, None, winSize=(21, 21), maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03,
            ),
        )
        if backward is None or backward_status is None:
            return Counter(), 0, []
        result = Counter()
        axis_points = 0
        vectors = []
        (x1, y1), (x2, y2) = geometry["line"]
        line_length = math.hypot(x2 - x1, y2 - y1)
        line_unit = ((x2 - x1) / line_length, (y2 - y1) / line_length)
        for start, end, returned, ok, backward_ok in zip(
            points.reshape(-1, 2), current.reshape(-1, 2),
            backward.reshape(-1, 2), status.ravel(), backward_status.ravel(),
        ):
            if not ok or not backward_ok or math.dist(start, returned) > 1.5:
                continue
            projection = (
                (end[0] - start[0]) * line_unit[0]
                + (end[1] - start[1]) * line_unit[1]
            )
            along_axis = (
                abs(projection) >= geometry["motion_min_displacement_px"]
            )
            if along_axis:
                axis_points += 1
            start_side = self._side(start, geometry["line"], MOTION_MARGIN_PX)
            end_side = self._side(end, geometry["line"], MOTION_MARGIN_PX)
            if not start_side or not end_side or start_side == end_side:
                if along_axis:
                    vectors.append((start, end, "axis"))
                continue
            direction = (
                "left_to_right" if (start_side, end_side) == (-1, 1)
                else "right_to_left"
            )
            direction = geometry["directions"][direction]
            result[direction] += 1
            vectors.append((start, end, direction))
        return result, axis_points, vectors

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
            ))

    def _match_motion(self, timestamp, force=False):
        for event in self.events:
            if event["confirmed"]:
                continue
            age = (timestamp - event["timestamp"]).total_seconds()
            if not force and age <= self.agreement_seconds:
                continue
            candidates = [
                candidate for candidate in self.motion_activity
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
            event["motion_candidate"] = candidate
            observation = event["observations"][0]
            self._diagnostic(
                "passage_agreement", timestamp,
                observation_ids=[observation["id"]],
                cameras=[observation["camera"], candidate["camera"]],
                direction=event["direction"],
                delta_seconds=round(delta, 3),
                motion_direction_points=candidate["direction_points"],
                motion_axis_points=candidate["axis_points"],
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

    def _save_ready_evidence(self):
        pending = []
        for kind, event_id, timestamp, cameras in self.pending_evidence:
            windows = {
                camera: self._evidence_window(camera, timestamp)
                for camera in cameras
            }
            if any(window is None for window in windows.values()):
                pending.append((kind, event_id, timestamp, cameras))
                continue
            for camera, window in windows.items():
                for offset, snapshot in zip(range(-3, 4), window):
                    self._save_evidence_frame(
                        kind, event_id, timestamp, camera, offset, snapshot,
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
        self, kind, event_id, timestamp, camera, offset, snapshot,
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
            for start, end, direction in labels:
                color = {
                    "entry": (0, 255, 0),
                    "exit": (0, 165, 255),
                    "axis": (255, 255, 0),
                }[direction]
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
        target = self.evidence_dir / (
            f"{kind}_{event_id}_{camera}_frame_{offset:+d}.jpg"
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
        confirmed = sum(event["confirmed"] for event in self.events)
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
            "confirmed_passages": sum(event["confirmed"] for event in self.events),
            "unconfirmed_passages": sum(
                not event["confirmed"] for event in self.events
            ),
            "passages": [
                self._passage_summary(event) for event in self.events
            ],
        }

    def _passage_summary(self, event):
        observation = event["observations"][0]
        motion = event.get("motion_candidate")
        return {
            "timestamp": self._timestamp(observation["timestamp"]),
            "direction": event["direction"],
            "camera": observation["camera"],
            "confirmed": event["confirmed"],
            "motion_timestamp": (
                self._timestamp(motion["timestamp"]) if motion else None
            ),
            "motion_delta_seconds": (
                round(abs((observation["timestamp"] - motion["timestamp"])
                          .total_seconds()), 3)
                if motion else None
            ),
        }

    def finish(self, timestamp):
        self._expire_events(timestamp, force=True)

    def close(self):
        if self.diagnostics:
            self.diagnostics.close()
            self.diagnostics = None
