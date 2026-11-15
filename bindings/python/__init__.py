"""Load this fork's native SQLite extension on an application connection."""

from pathlib import Path
import sqlite3

from .extra_init import serialize_float32, serialize_int8

__all__ = ["load", "get_loadable_path", "serialize_float32", "serialize_int8"]


def get_loadable_path() -> str:
    """Return the extension path without its platform suffix, as SQLite expects."""
    return str(Path(__file__).resolve().with_name("vec0"))


def load(connection: sqlite3.Connection) -> None:
    """Load vec0 after the caller enables extension loading on this connection."""
    connection.load_extension(get_loadable_path())
