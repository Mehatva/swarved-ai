from config import DATA_DIR, DATABASE_PATH
from typing import Any

import aiosqlite

CREATE_INCIDENTS_TABLE = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    caller_id TEXT NOT NULL,
    receiver_device_hash TEXT NOT NULL,
    synthetic_voice_probability REAL NOT NULL,
    audio_fingerprint_hash TEXT NOT NULL,
    gps_lat REAL NOT NULL,
    gps_lon REAL NOT NULL,
    threat_category TEXT NOT NULL,
    i4c_reference_number TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


async def initialize_database() -> None:
    """
    Create the database directory and incidents table if they don't exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DATABASE_PATH) as connection:
        await connection.execute("PRAGMA journal_mode=WAL;")
        await connection.execute(CREATE_INCIDENTS_TABLE)
        await connection.commit()


async def insert_incident(incident: dict[str, Any]) -> None:
    """
    Persist an incident in SQLite.
    """
    query = """
    INSERT INTO incidents (
        incident_id,
        timestamp,
        caller_id,
        receiver_device_hash,
        synthetic_voice_probability,
        audio_fingerprint_hash,
        gps_lat,
        gps_lon,
        threat_category,
        i4c_reference_number
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        incident["incident_id"],
        incident["timestamp"],
        incident["caller_id"],
        incident["receiver_device_hash"],
        incident["synthetic_voice_probability"],
        incident["audio_fingerprint_hash"],
        incident["gps_lat"],
        incident["gps_lon"],
        incident["threat_category"],
        incident["i4c_reference_number"],
    )

    async with aiosqlite.connect(DATABASE_PATH) as connection:
        await connection.execute(query, values)
        await connection.commit()