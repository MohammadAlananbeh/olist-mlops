import sqlite3
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "olist.db"

# Ensure the database directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_prediction(
    prediction: int,
    probability: float,
    model_version: str,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            INSERT INTO prediction_logs
            (timestamp, prediction, probability, model_version)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                prediction,
                probability,
                model_version,
            ),
        )

        conn.commit()
