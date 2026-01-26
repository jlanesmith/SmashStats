#!/usr/bin/env python3
"""Restore database from backup and reorder by datetime."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "smashstats.db"
BACKUP_PATH = Path(__file__).parent / "database_data.txt"


def restore_and_reorder():
    # Read backup file
    with open(BACKUP_PATH, 'r') as f:
        lines = f.readlines()

    # Skip the first line (command prompt)
    data_lines = [line.strip() for line in lines[1:] if line.strip()]

    print(f"Found {len(data_lines)} records in backup")

    # Parse records
    records = []
    for line in data_lines:
        parts = line.split('|')
        if len(parts) >= 16:
            # id|datetime|p1_char|p2_char|p3_char|p4_char|p1_kos|p2_kos|p3_kos|p4_kos|p1_dmg|p2_dmg|p3_dmg|p4_dmg|win|opponent
            record = {
                'datetime': parts[1],
                'p1_character': parts[2],
                'p2_character': parts[3],
                'p3_character': parts[4],
                'p4_character': parts[5],
                'p1_kos': int(parts[6]) if parts[6] else None,
                'p2_kos': int(parts[7]) if parts[7] else None,
                'p3_kos': int(parts[8]) if parts[8] else None,
                'p4_kos': int(parts[9]) if parts[9] else None,
                'p1_damage': int(parts[10]) if parts[10] else None,
                'p2_damage': int(parts[11]) if parts[11] else None,
                'p3_damage': int(parts[12]) if parts[12] else None,
                'p4_damage': int(parts[13]) if parts[13] else None,
                'win': parts[14],
                'opponent': parts[15] if len(parts) > 15 else ''
            }
            records.append(record)

    print(f"Parsed {len(records)} records")

    # Sort by datetime ascending (oldest first)
    records.sort(key=lambda x: x['datetime'])

    print(f"Sorted records by datetime")
    print(f"  Oldest: {records[0]['datetime']}")
    print(f"  Newest: {records[-1]['datetime']}")

    # Recreate database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop and recreate table
    cursor.execute("DROP TABLE IF EXISTS games")
    cursor.execute("""
        CREATE TABLE games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT,
            p1_character TEXT,
            p2_character TEXT,
            p3_character TEXT,
            p4_character TEXT,
            p1_kos INT,
            p2_kos INT,
            p3_kos INT,
            p4_kos INT,
            p1_damage INT,
            p2_damage INT,
            p3_damage INT,
            p4_damage INT,
            win TEXT,
            opponent TEXT
        )
    """)

    # Insert records in sorted order
    for record in records:
        cursor.execute("""
            INSERT INTO games (datetime, p1_character, p2_character, p3_character, p4_character,
                p1_kos, p2_kos, p3_kos, p4_kos, p1_damage, p2_damage, p3_damage, p4_damage, win, opponent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record['datetime'], record['p1_character'], record['p2_character'],
            record['p3_character'], record['p4_character'],
            record['p1_kos'], record['p2_kos'], record['p3_kos'], record['p4_kos'],
            record['p1_damage'], record['p2_damage'], record['p3_damage'], record['p4_damage'],
            record['win'], record['opponent']
        ))

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]
    print(f"\nRestored {count} games to database")

    cursor.execute("SELECT id, datetime FROM games LIMIT 5")
    print("\nFirst 5 (oldest):")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cursor.execute("SELECT id, datetime FROM games ORDER BY id DESC LIMIT 5")
    print("\nLast 5 (newest):")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    conn.close()


if __name__ == "__main__":
    restore_and_reorder()
