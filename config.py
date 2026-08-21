"""Project configuration."""

import tomllib
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
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


def frame_interval_ms(config_path=CONFIG_PATH):
    value = _settings(config_path).get("frame_interval_ms", 333)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("config frame_interval_ms must be a positive integer")
    return value


def playback_speed(config_path=CONFIG_PATH):
    value = _settings(config_path).get("playback_speed", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("config playback_speed must be greater than zero")
    return float(value)
