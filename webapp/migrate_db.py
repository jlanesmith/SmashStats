#!/usr/bin/env python3
"""
Migration script to update database schema.
Run this to allow NULL values for all player stats (KOs, falls, damage).
"""

import sys
from pathlib import Path

# Add webapp directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from database import migrate_to_nullable_stats

if __name__ == "__main__":
    print("=" * 50)
    print("SmashStats Database Migration")
    print("=" * 50)
    print("\nThis will update your database to allow NULL values")
    print("for all player stats (KOs, falls, damage).\n")

    response = input("Continue? (y/n): ").strip().lower()
    if response == 'y':
        migrate_to_nullable_stats()
        print("\n✓ Migration successful!")
    else:
        print("Migration cancelled.")
