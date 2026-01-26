#!/usr/bin/env python3
"""
Migrate datetime format from M/D/YYYY HH:MM:SS to YYYY-MM-DD HH:MM:SS.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "smashstats.db"


def migrate_datetime_format():
    """Convert old datetime format to new ISO format."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find all records with old format (contains "/")
    cursor.execute("SELECT id, datetime FROM games WHERE datetime LIKE '%/%'")
    rows = cursor.fetchall()

    print(f"Found {len(rows)} records with old datetime format")

    updated = 0
    errors = []

    for game_id, old_datetime in rows:
        try:
            # Parse old format: M/D/YYYY HH:MM:SS
            dt = datetime.strptime(old_datetime, "%m/%d/%Y %H:%M:%S")
            # Convert to new format: YYYY-MM-DD HH:MM:SS
            new_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                "UPDATE games SET datetime = ? WHERE id = ?",
                (new_datetime, game_id)
            )
            updated += 1

        except ValueError as e:
            errors.append(f"ID {game_id}: {old_datetime} - {e}")

    conn.commit()
    conn.close()

    print(f"Updated {updated} records")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors[:10]:
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    print(f"Database: {DB_PATH}")
    migrate_datetime_format()

    # Verify
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM games WHERE datetime LIKE '%/%'")
    remaining = cursor.fetchone()[0]
    print(f"\nRemaining old format records: {remaining}")

    cursor.execute("SELECT datetime FROM games ORDER BY datetime DESC LIMIT 5")
    print("\nSample dates after migration:")
    for row in cursor.fetchall():
        print(f"  {row[0]}")
    conn.close()
