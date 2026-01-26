#!/usr/bin/env python3
"""
Sync local SmashStats database to remote Railway API.
Uploads all local games that don't exist on the remote server.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from webapp.database import get_all_games


def sync_games(remote_url: str, dry_run: bool = False) -> Tuple[int, int]:
    """
    Sync all local games to the remote server.

    Args:
        remote_url: The base URL of the remote server
        dry_run: If True, don't actually upload, just show what would be done

    Returns:
        tuple: (games_synced, games_failed)
    """
    api_url = f"{remote_url.rstrip('/')}/api/games"

    # Get all local games
    local_games = get_all_games()
    print(f"Found {len(local_games)} local games")

    # Get remote games to check for duplicates
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        remote_data = response.json()
        remote_games = remote_data.get("games", [])
        print(f"Found {len(remote_games)} remote games")
    except Exception as e:
        print(f"Warning: Could not fetch remote games: {e}")
        print("Will upload all games (may create duplicates if some already exist)")
        remote_games = []

    # Create a set of remote game identifiers (datetime + characters + win)
    remote_keys = set()
    for g in remote_games:
        key = (
            g.get("datetime", ""),
            g.get("p1_character", ""),
            g.get("p2_character", ""),
            g.get("p3_character", ""),
            g.get("p4_character", ""),
            g.get("win", "")
        )
        remote_keys.add(key)

    # Filter to games that don't exist remotely
    games_to_sync = []
    for g in local_games:
        key = (
            g.get("datetime", ""),
            g.get("p1_character", ""),
            g.get("p2_character", ""),
            g.get("p3_character", ""),
            g.get("p4_character", ""),
            g.get("win", "")
        )
        if key not in remote_keys:
            games_to_sync.append(g)

    print(f"Games to sync: {len(games_to_sync)}")

    if dry_run:
        print("\n[DRY RUN] Would upload the following games:")
        for g in games_to_sync[:10]:
            print(f"  - {g['datetime']}: {g['p1_character']} & {g['p2_character']} vs {g['p3_character']} & {g['p4_character']} -> {g['win']}")
        if len(games_to_sync) > 10:
            print(f"  ... and {len(games_to_sync) - 10} more")
        return len(games_to_sync), 0

    # Upload games (oldest first to maintain order)
    games_to_sync.sort(key=lambda g: g.get("datetime", ""))

    synced = 0
    failed = 0

    for i, game in enumerate(games_to_sync):
        # Remove the local ID - remote will assign its own
        game_data = {k: v for k, v in game.items() if k != "id"}

        try:
            response = requests.post(api_url, json=game_data, timeout=10)
            if response.status_code == 200:
                synced += 1
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i + 1}/{len(games_to_sync)} games synced")
            else:
                failed += 1
                print(f"  Failed to sync game {game['datetime']}: status {response.status_code}")
        except Exception as e:
            failed += 1
            print(f"  Failed to sync game {game['datetime']}: {e}")

    return synced, failed


def main():
    parser = argparse.ArgumentParser(
        description="Sync local SmashStats database to remote Railway server"
    )
    parser.add_argument(
        "remote_url",
        nargs="?",
        default="https://smashstats.up.railway.app",
        help="Remote server URL (default: https://smashstats.up.railway.app)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually uploading"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("SMASHSTATS REMOTE SYNC")
    print("=" * 50)
    print(f"\nRemote URL: {args.remote_url}")
    print()

    synced, failed = sync_games(args.remote_url, dry_run=args.dry_run)

    print()
    print("=" * 50)
    if args.dry_run:
        print(f"[DRY RUN] Would sync {synced} games")
    else:
        print(f"Synced: {synced} games")
        if failed > 0:
            print(f"Failed: {failed} games")
    print("=" * 50)


if __name__ == "__main__":
    main()
