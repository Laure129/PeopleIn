"""Project configuration."""

import tomllib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.toml"
DATABASE_PATH = PROJECT_DIR / "data" / "read_files.sqlite3"


def _settings(config_path=CONFIG_PATH):
    path = Path(config_path)
    return (
        tomllib.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {}
    )


def debug_mode(config_path=CONFIG_PATH):
    """Return the configured debug flag; missing config defaults to false."""
    debug = _settings(config_path).get("debug", False)
    if not isinstance(debug, bool):
        raise ValueError("config debug must be true or false")
    return debug


def prepare_benchmark_enabled(config_path=CONFIG_PATH):
    """Return whether competing processes should be stopped before a run."""
    enabled = _settings(config_path).get("prepare_benchmark", False)
    if not isinstance(enabled, bool):
        raise ValueError("config prepare_benchmark must be true or false")
    return enabled


def frame_interval_ms(config_path=CONFIG_PATH):
    value = _settings(config_path).get("frame_interval_ms", 333)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("config frame_interval_ms must be a positive integer")
    return value


def archive_dir(config_path=CONFIG_PATH):
    value = _settings(config_path).get("archive_dir", "archive_debug_cache")
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or Path(value).name != value
    ):
        raise ValueError("config archive_dir must be a resource directory name")
    return PROJECT_DIR / "resources" / value


def stream_cameras(config_path=CONFIG_PATH):
    cameras = _settings(config_path).get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("config cameras must be a table")

    selected = {}
    for camera, values in cameras.items():
        if not isinstance(values, dict):
            raise ValueError(f"config camera {camera} must be a table")
        view = values.get("view")
        if view not in {"entrance", "exit"}:
            continue
        if view in selected:
            raise ValueError(f"config has multiple cameras with view {view}")
        selected[view] = camera

    if set(selected) != {"entrance", "exit"}:
        raise ValueError("config must define entrance and exit camera views")
    return selected["entrance"], selected["exit"]


def door_counter_settings(config_path=CONFIG_PATH):
    path = Path(config_path)
    settings = _settings(path)
    model = settings.get("person_model")
    confidence = settings.get("person_confidence")
    agreement = settings.get("door_agreement_seconds")
    margin = settings.get("crossing_margin_px")
    if not isinstance(model, str) or not model or Path(model).is_absolute():
        raise ValueError("config person_model must be a relative path")
    if ".." in Path(model).parts:
        raise ValueError("config person_model must stay inside the project")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
            or not 0 < confidence < 1:
        raise ValueError("config person_confidence must be between zero and one")
    if isinstance(agreement, bool) or not isinstance(agreement, (int, float)) \
            or agreement <= 0:
        raise ValueError("config door_agreement_seconds must be positive")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) \
            or margin < 0:
        raise ValueError("config crossing_margin_px must not be negative")

    cameras = settings.get("cameras")
    result = {}
    for camera in stream_cameras(path):
        values = cameras[camera]
        line = values.get("door_line")
        if (
            not isinstance(line, list) or len(line) != 2
            or any(
                not isinstance(point, list) or len(point) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in point
                )
                for point in line
            )
            or line[0] == line[1]
        ):
            raise ValueError(f"config camera {camera} door_line is invalid")
        directions = {
            values.get("entry_direction"): "entry",
            values.get("exit_direction"): "exit",
        }
        if set(directions) != {"left_to_right", "right_to_left"}:
            raise ValueError(f"config camera {camera} directions are invalid")
        door_confidence = values.get("door_confidence", confidence)
        radius = values.get("door_confidence_radius_px", 0)
        if isinstance(door_confidence, bool) or not isinstance(
            door_confidence, (int, float)
        ) or not 0 < door_confidence <= confidence:
            raise ValueError(
                f"config camera {camera} door_confidence is invalid"
            )
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) \
                or radius < 0:
            raise ValueError(
                f"config camera {camera} door_confidence_radius_px is invalid"
            )
        result[camera] = {
            "line": tuple(tuple(point) for point in line),
            "directions": directions,
            "door_confidence": float(door_confidence),
            "door_confidence_radius_px": float(radius),
        }
        motion_roi = values.get("motion_roi")
        if motion_roi is not None:
            if (
                not isinstance(motion_roi, list) or len(motion_roi) != 2
                or any(
                    not isinstance(point, list) or len(point) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in point
                    )
                    for point in motion_roi
                )
                or motion_roi[0][0] >= motion_roi[1][0]
                or motion_roi[0][1] >= motion_roi[1][1]
            ):
                raise ValueError(f"config camera {camera} motion_roi is invalid")
            minimum = values.get("motion_min_area")
            if isinstance(minimum, bool) or not isinstance(minimum, int) \
                    or minimum <= 0:
                raise ValueError(
                    f"config camera {camera} motion_min_area is invalid"
                )
            result[camera]["motion_roi"] = tuple(
                tuple(point) for point in motion_roi
            )
            result[camera]["motion_min_area"] = minimum
    return {
        "model_path": path.resolve().parent / model,
        "confidence": float(confidence),
        "agreement_seconds": float(agreement),
        "crossing_margin_px": float(margin),
        "cameras": result,
    }
