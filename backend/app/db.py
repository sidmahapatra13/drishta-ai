"""SQLite persistence.

SQLite rather than PostgreSQL deliberately: it is a single file with no server
to run, no container to build and no credentials to distribute, which matters
when five teammates need the backend running on their laptops tomorrow morning.
The schema mirrors the one in the master README so a Postgres migration later
is mechanical.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "local-data" / "screening.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_ref TEXT PRIMARY KEY,
    age INTEGER,
    sex TEXT,
    diabetes_duration_years REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS screenings (
    screening_id TEXT PRIMARY KEY,
    patient_ref TEXT NOT NULL REFERENCES patients(patient_ref),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    result_json TEXT NOT NULL,
    grade INTEGER,
    referable INTEGER,
    priority TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id TEXT NOT NULL REFERENCES screenings(screening_id),
    reviewer_ref TEXT NOT NULL,
    action TEXT NOT NULL,
    override_grade INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_ref);
CREATE INDEX IF NOT EXISTS idx_screenings_review ON screenings(review_status, priority);
"""


@contextmanager
def session():
    """Open a connection, commit on success, always close.

    sqlite3's own context manager commits but does not close, which leaks a
    file handle per request. This wrapper does both.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init() -> None:
    """Idempotent. Called at import time so the schema always exists before the
    first request, rather than depending on a startup hook that test clients
    and some ASGI servers do not fire."""
    with session() as conn:
        conn.executescript(SCHEMA)


def upsert_patient(conn, req) -> None:
    conn.execute(
        """INSERT INTO patients (patient_ref, age, sex, diabetes_duration_years)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(patient_ref) DO UPDATE SET
             age = excluded.age,
             sex = excluded.sex,
             diabetes_duration_years = excluded.diabetes_duration_years""",
        (req.patient_ref, req.age, req.sex, req.diabetes_duration_years),
    )


def save_screening(conn, result) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO screenings
           (screening_id, patient_ref, result_json, grade, referable, priority)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            result.screening_id,
            result.patient_ref,
            result.model_dump_json(),
            result.grade,
            int(result.referable) if result.referable is not None else None,
            result.priority.value if result.priority else None,
        ),
    )


def get_screening(conn, screening_id: str) -> dict | None:
    row = conn.execute(
        "SELECT result_json FROM screenings WHERE screening_id = ?", (screening_id,)
    ).fetchone()
    return json.loads(row["result_json"]) if row else None


def patient_history(conn, patient_ref: str) -> list[dict]:
    rows = conn.execute(
        """SELECT screening_id, created_at, grade, referable, priority, review_status
           FROM screenings WHERE patient_ref = ? ORDER BY created_at DESC""",
        (patient_ref,),
    ).fetchall()
    return [dict(r) for r in rows]


def review_queue(conn) -> list[dict]:
    """Referable and ungradable cases first, most severe at the top.

    Ordering is the product feature here: the point of the system is that a
    scarce ophthalmologist sees the urgent cases first.
    """
    rows = conn.execute(
        """SELECT screening_id, patient_ref, created_at, grade, referable,
                  priority, review_status
           FROM screenings
           ORDER BY CASE priority WHEN 'urgent' THEN 0 WHEN 'review' THEN 1
                                  ELSE 2 END,
                    grade IS NULL, grade DESC, created_at ASC"""
    ).fetchall()
    return [dict(r) for r in rows]


def save_review(conn, screening_id: str, action) -> None:
    conn.execute(
        """INSERT INTO reviews (screening_id, reviewer_ref, action, override_grade, notes)
           VALUES (?, ?, ?, ?, ?)""",
        (screening_id, action.reviewer_ref, action.action, action.override_grade, action.notes),
    )
    conn.execute(
        "UPDATE screenings SET review_status = ? WHERE screening_id = ?",
        ("reviewed" if action.action != "request_recapture" else "recapture", screening_id),
    )
