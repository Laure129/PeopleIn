"""Project configuration."""

import tomllib
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.toml"
DATABASE_PATH = PROJECT_DIR / "data" / "read_files.sqlite3"


def debug_mode(config_path=CONFIG_PATH):
    """Return the configured debug flag; missing config defaults to false."""
    path = Path(config_path)
    config = (
        tomllib.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {}
    )
    debug = config.get("debug", False)
    if not isinstance(debug, bool):
        raise ValueError("config debug must be true or false")
    return debug
