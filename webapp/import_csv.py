#!/usr/bin/env python3
"""
Import existing CSV data into the SmashStats database.
"""

import csv
import sys
from pathlib import Path

# Add parent directory to path to import database module
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, save_game_result, get_game_count, DB_PATH


def import_csv(csv_path: str, skip_existing: bool = True):
    """
    Import games from a CSV file into the database.

    Args:
        csv_path: Path to the CSV file
        skip_existing: If True, skip import if database already has data
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    # Initialize database
    init_db()

    # Check if database already has data
    existing_count = get_game_count()
    if existing_count > 0 and skip_existing:
        print(f"Database already contains {existing_count} games.")
        print("Use --force to reimport (will add duplicates).")
        return

    # Read and import CSV
    imported = 0
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Build result dict with proper types
            result = {
                'datetime': row['datetime'],
                'win': row['win'],
                'opponent': row.get('opponent', ''),
                'p1_character': row['p1_character'],
                'p2_character': row['p2_character'],
                'p3_character': row['p3_character'],
                'p4_character': row['p4_character'],
                'p1_kos': int(row['p1_kos']) if row['p1_kos'] else 0,
                'p2_kos': int(row['p2_kos']) if row['p2_kos'] else 0,
                'p3_kos': int(row['p3_kos']) if row['p3_kos'] else 0,
                'p4_kos': int(row['p4_kos']) if row['p4_kos'] else 0,
                'p1_falls': int(row['p1_falls']) if row['p1_falls'] else 0,
                'p2_falls': int(row['p2_falls']) if row['p2_falls'] else 0,
                'p3_falls': int(row['p3_falls']) if row['p3_falls'] else 0,
                'p4_falls': int(row['p4_falls']) if row['p4_falls'] else 0,
                'p1_damage': int(row['p1_damage']) if row['p1_damage'] else 0,
                'p2_damage': int(row['p2_damage']) if row['p2_damage'] else 0,
                'p3_damage': int(row['p3_damage']) if row['p3_damage'] else 0,
                'p4_damage': int(row['p4_damage']) if row['p4_damage'] else 0,
            }

            game_id = save_game_result(result)
            imported += 1
            print(f"Imported game {game_id}: {result['datetime']} - {result['p1_character']} vs {result['p3_character']}")

    print(f"\nImported {imported} games from {csv_path}")
    print(f"Total games in database: {get_game_count()}")


def clear_database():
    """Clear all data from the database."""
    from database import get_connection

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM games")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'games'")
    conn.commit()
    conn.close()
    print("Database cleared.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import CSV data into SmashStats database")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(Path(__file__).parent.parent / "results.csv"),
        help="Path to CSV file (default: results.csv)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Import even if database already has data (may create duplicates)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear database before importing"
    )

    args = parser.parse_args()

    if args.clear:
        clear_database()

    import_csv(args.csv_path, skip_existing=not args.force)
