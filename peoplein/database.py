"""Track successfully read archive files in SQLite."""

import sqlite3
from contextlib import closing
from pathlib import Path


def _connect(database_path):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS read_files ("
        "path TEXT PRIMARY KEY, "
        "read_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    connection.commit()
    return connection


def ensure_unread(database_path, paths):
    required = set(paths)
    with closing(_connect(database_path)) as connection:
        already_read = {
            row[0] for row in connection.execute("SELECT path FROM read_files")
        }
    duplicates = sorted(required & already_read)
    if duplicates:
        raise RuntimeError("archive files already read: " + ", ".join(duplicates))


def mark_read(database_path, path):
    with closing(sqlite3.connect(database_path, timeout=30)) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO read_files(path) VALUES (?)", (path,),
        )
        connection.commit()
