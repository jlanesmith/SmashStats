#!/usr/bin/env python3
"""
Import game data from CSV file into the database.
"""

import csv
import sys
from pathlib import Path
from webapp.database import save_game_result, get_game_count, init_db

def import_csv(csv_path: str):
    """Import games from CSV file."""

    # Initialize database if needed
    init_db()

    initial_count = get_game_count()
    print(f"Starting import from: {csv_path}")
    print(f"Current game count: {initial_count}")

    imported = 0
    errors = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=['time', 'p1', 'p2', 'p3', 'p4', 'win'])

        for row_num, row in enumerate(reader, 1):
            try:
                # Convert CSV row to database format
                result = {
                    'datetime': row['time'].strip(),
                    'p1_character': row['p1'].strip(),
                    'p2_character': row['p2'].strip(),
                    'p3_character': row['p3'].strip(),
                    'p4_character': row['p4'].strip(),
                    'win': row['win'].strip(),
                    # Optional fields are not in CSV, will be NULL
                    'p1_kos': None,
                    'p2_kos': None,
                    'p3_kos': None,
                    'p4_kos': None,
                    'p1_damage': None,
                    'p2_damage': None,
                    'p3_damage': None,
                    'p4_damage': None,
                    'opponent': ''
                }

                # Validate win value
                if result['win'] not in ['Yes', 'No']:
                    raise ValueError(f"Invalid win value: {result['win']}")

                # Save to database
                game_id = save_game_result(result)
                imported += 1

                if imported % 100 == 0:
                    print(f"Imported {imported} games...")

            except Exception as e:
                error_msg = f"Row {row_num}: {str(e)} - {row}"
                errors.append(error_msg)
                print(f"ERROR: {error_msg}")

    final_count = get_game_count()
    print("\n" + "="*50)
    print(f"Import complete!")
    print(f"Successfully imported: {imported} games")
    print(f"Errors: {len(errors)}")
    print(f"Final game count: {final_count} (was {initial_count})")

    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python import_csv.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    import_csv(csv_path)
