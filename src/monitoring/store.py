"""SQLite helpers for drift monitoring persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = str(ROOT_DIR / "backend" / "product_categorization.db")


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Ensure monitoring tables exist."""
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            embedding_score REAL,
            confidence_score REAL,
            class_score REAL,
            is_drift INTEGER NOT NULL DEFAULT 0,
            embedding_drifted INTEGER NOT NULL DEFAULT 0,
            confidence_drifted INTEGER NOT NULL DEFAULT 0,
            class_drifted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            alert_type TEXT NOT NULL,
            message TEXT,
            resolved INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def insert_drift_event(
    embedding_score: float,
    confidence_score: float,
    class_score: float,
    is_drift: bool,
    embedding_drifted: bool,
    confidence_drifted: bool,
    class_drifted: bool,
) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO drift_events (
            timestamp,
            embedding_score,
            confidence_score,
            class_score,
            is_drift,
            embedding_drifted,
            confidence_drifted,
            class_drifted
        )
        VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            float(embedding_score),
            float(confidence_score),
            float(class_score),
            int(bool(is_drift)),
            int(bool(embedding_drifted)),
            int(bool(confidence_drifted)),
            int(bool(class_drifted)),
        ),
    )
    conn.commit()
    conn.close()


def insert_alert(alert_type: str, message: str) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO alerts (timestamp, alert_type, message, resolved)
        VALUES (CURRENT_TIMESTAMP, ?, ?, 0)
        """,
        (alert_type, message),
    )
    conn.commit()
    conn.close()
