"""Detect people and count directed door-line crossings."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .database import record_passage


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
        return [boxes[int(index)] for index in np.asarray(indices).reshape(-1)]


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
    ):
        self.cameras = cameras
        self.detector = detector or PersonDetector(model_path, confidence)
        self.agreement_seconds = agreement_seconds
        self.crossing_margin_px = crossing_margin_px
        self.database_path = database_path
        self.app_version = app_version
        self.tracks = {camera: {} for camera in cameras}
        self.next_track_id = 1
        self.events = []

    def update(self, camera, frame, timestamp):
        geometry = self.cameras[camera]
        line = geometry["line"]
        feet = [
            (x + width / 2, y + height)
            for x, y, width, height in self.detector(frame)
        ]
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
        for distance, track_id, index in matches:
            if distance > 120 or track_id in used_tracks or index in used_feet:
                continue
            used_tracks.add(track_id)
            used_feet.add(index)
            direction = self._move_track(tracks[track_id], feet[index], line)
            if direction is not None:
                self._passage(
                    timestamp, camera, geometry["directions"][direction],
                )

        for index, foot in enumerate(feet):
            if index in used_feet:
                continue
            side = self._side(foot, line)
            tracks[self.next_track_id] = Track(foot, side, foot)
            self.next_track_id += 1
        for track_id in [
            track_id for track_id, track in tracks.items() if track.misses > 5
        ]:
            del tracks[track_id]

    def _move_track(self, track, foot, line):
        track.misses = 0
        track.foot = foot
        side = self._side(foot, line)
        if side == 0:
            return None
        if track.side == 0 or side == track.side:
            track.side = side
            track.side_point = foot
            return None
        previous_side = track.side
        crossed = self._segments_intersect(track.side_point, foot, *line)
        track.side = side
        track.side_point = foot
        if not crossed:
            return None
        return (
            "left_to_right" if (previous_side, side) == (-1, 1)
            else "right_to_left"
        )

    def _side(self, point, line):
        (x1, y1), (x2, y2) = line
        distance = (
            (point[0] - x1) * (y2 - y1)
            - (point[1] - y1) * (x2 - x1)
        ) / math.hypot(x2 - x1, y2 - y1)
        if distance < -self.crossing_margin_px:
            return -1
        if distance > self.crossing_margin_px:
            return 1
        return 0

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

    def _passage(self, timestamp, camera, direction):
        record_passage(
            self.database_path, timestamp, self.app_version, camera, direction,
        )
        # ponytail: time-only agreement; add cross-camera ReID if simultaneous
        # passages prove that this pairs different people.
        matches = [
            event for event in self.events
            if event["direction"] == direction
            and camera not in event["cameras"]
            and abs((timestamp - event["timestamp"]).total_seconds())
            <= self.agreement_seconds
        ]
        if matches:
            event = min(
                matches,
                key=lambda item: abs((timestamp - item["timestamp"]).total_seconds()),
            )
            event["cameras"].add(camera)
        else:
            self.events.append({
                "timestamp": timestamp,
                "direction": direction,
                "cameras": {camera},
            })

    def snapshot(self):
        entered = sum(event["direction"] == "entry" for event in self.events)
        exited = sum(event["direction"] == "exit" for event in self.events)
        confirmed = sum(len(event["cameras"]) > 1 for event in self.events)
        confidence = confirmed / len(self.events) if self.events else 1.0
        return {
            "entered_total": entered,
            "exited_total": exited,
            "people_inside": max(entered - exited, 0),
            "people_inside_confidence": round(confidence, 6),
        }
