"""SQLite connection management and migrations."""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2
SCHEMA_SQL = Path(__file__).with_name("schema.sql")
MIGRATIONS_DIR = Path(__file__).with_name("migrations")

_local = threading.local()


def _configure(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    # Indexing writes from several workers; without this they fail instantly
    # instead of waiting their turn behind the write lock.
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")


def connect(db_path: Path) -> sqlite3.Connection:
    """Return this thread's connection, opening it on first use."""
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != db_path:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None disables the legacy sqlite3 driver's habit of
        # opening an implicit transaction before every INSERT/UPDATE, which
        # collides with our explicit BEGIN IMMEDIATE. Transaction boundaries
        # are now exactly where transaction() puts them.
        conn = sqlite3.connect(db_path, timeout=15.0, check_same_thread=False,
                               isolation_level=None)
        _configure(conn)
        _local.conn = conn
        _local.path = db_path
    return conn


def _migrate(conn: sqlite3.Connection, current: int) -> int:
    """Apply numbered migrations above `current`, in order.

    schema.sql is the shape of a *new* database; these carry an existing one
    forward. Each file runs inside its own transaction, so a failure leaves the
    version untouched and the migration is retried next start rather than
    leaving the database half-upgraded.
    """
    if not MIGRATIONS_DIR.is_dir():
        return current
    pending = sorted(
        (int(p.name.split("_", 1)[0]), p)
        for p in MIGRATIONS_DIR.glob("*.sql")
        if p.name.split("_", 1)[0].isdigit()
    )
    for version, path in pending:
        if version <= current:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executescript(path.read_text())
            conn.execute("UPDATE schema_version SET version=?", (version,))
        except Exception:
            conn.rollback()
            raise
        conn.commit()
        current = version
    return current


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    fresh = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='photo'"
    ).fetchone()[0] == 0
    conn.executescript(SCHEMA_SQL.read_text())
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        # A brand new database already has the latest shape from schema.sql;
        # an older one that predates versioning starts at 1.
        conn.execute("INSERT INTO schema_version (version) VALUES (?)",
                     (SCHEMA_VERSION if fresh else 1,))
        conn.commit()
        current = SCHEMA_VERSION if fresh else 1
    else:
        current = row["version"]
    _migrate(conn, current)
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit IMMEDIATE transaction: takes the write lock up front so two
    workers cannot both read, both decide to insert, and one lose."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def close() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
