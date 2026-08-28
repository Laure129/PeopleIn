"""Project configuration."""

import os
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.toml"
DATABASE_PATH = PROJECT_DIR / "data" / "read_files.sqlite3"
ARCHIVE_ENV_KEYS = {
    "ARCHIVE_BASE_URL", "ARCHIVE_LOGIN", "ARCHIVE_PASSWORD",
}


def _load_archive_env(path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in ARCHIVE_ENV_KEYS or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


_load_archive_env(PROJECT_DIR / ".env")


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


def motion_frame_interval_ms(config_path=CONFIG_PATH):
    value = _settings(config_path).get(
        "motion_frame_interval_ms", frame_interval_ms(config_path),
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "config motion_frame_interval_ms must be a positive integer"
        )
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


def archive_server():
    """Return the optional remote archive and its Basic Auth credentials."""
    base_url = os.getenv("ARCHIVE_BASE_URL", "").strip()
    login = os.getenv("ARCHIVE_LOGIN", "")
    password = os.getenv("ARCHIVE_PASSWORD", "")
    if not base_url:
        return None
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "ARCHIVE_BASE_URL must be an HTTP(S) URL without credentials"
        )
    return base_url.rstrip("/"), login, password


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
            band_width = values.get("motion_band_width_px")
            minimum = values.get("motion_min_points")
            displacement = values.get("motion_min_displacement_px")
            perpendicular_only = values.get(
                "motion_perpendicular_only", False,
            )
            profile_displacement = values.get(
                "motion_profile_min_displacement_px", displacement,
            )
            profile_minimum = values.get(
                "motion_profile_min_points", minimum,
            )
            profile_open_minimum = values.get(
                "motion_profile_open_min_points", minimum,
            )
            if isinstance(band_width, bool) or not isinstance(band_width, int) \
                    or band_width <= 0:
                raise ValueError(
                    f"config camera {camera} motion_band_width_px is invalid"
                )
            if isinstance(minimum, bool) or not isinstance(minimum, int) \
                    or minimum <= 0:
                raise ValueError(
                    f"config camera {camera} motion_min_points is invalid"
                )
            if isinstance(displacement, bool) or not isinstance(
                displacement, (int, float)
            ) or displacement <= 0:
                raise ValueError(
                    f"config camera {camera} "
                    "motion_min_displacement_px is invalid"
                )
            if not isinstance(perpendicular_only, bool):
                raise ValueError(
                    f"config camera {camera} "
                    "motion_perpendicular_only must be true or false"
                )
            if isinstance(profile_displacement, bool) or not isinstance(
                profile_displacement, (int, float)
            ) or profile_displacement <= 0:
                raise ValueError(
                    f"config camera {camera} "
                    "motion_profile_min_displacement_px is invalid"
                )
            if isinstance(profile_minimum, bool) or not isinstance(
                profile_minimum, int
            ) or profile_minimum <= 0:
                raise ValueError(
                    f"config camera {camera} "
                    "motion_profile_min_points is invalid"
                )
            if isinstance(profile_open_minimum, bool) or not isinstance(
                profile_open_minimum, int
            ) or profile_open_minimum <= 0:
                raise ValueError(
                    f"config camera {camera} "
                    "motion_profile_open_min_points is invalid"
                )
            result[camera]["motion_roi"] = tuple(
                tuple(point) for point in motion_roi
            )
            result[camera]["motion_band_width_px"] = band_width
            result[camera]["motion_min_points"] = minimum
            result[camera]["motion_min_displacement_px"] = float(displacement)
            result[camera]["motion_perpendicular_only"] = perpendicular_only
            result[camera]["motion_profile_min_displacement_px"] = float(
                profile_displacement
            )
            result[camera]["motion_profile_min_points"] = profile_minimum
            result[camera][
                "motion_profile_open_min_points"
            ] = profile_open_minimum
    return {
        "model_path": path.resolve().parent / model,
        "confidence": float(confidence),
        "agreement_seconds": float(agreement),
        "crossing_margin_px": float(margin),
        "cameras": result,
    }
