#!/usr/bin/env python3
"""
Database module for SmashStats.
Uses SQLite for local storage of game results.
"""

import sqlite3
from pathlib import Path

# Database path - stored in the project root
DB_PATH = Path(__file__).parent.parent / "smashstats.db"


def get_connection():
    """Get a database connection with row factory for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create single games table with all player stats
    # Stats can be NULL (displayed as N/A in UI)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            p1_character TEXT,
            p2_character TEXT,
            p3_character TEXT,
            p4_character TEXT,
            p1_kos INTEGER,
            p2_kos INTEGER,
            p3_kos INTEGER,
            p4_kos INTEGER,
            p1_falls INTEGER,
            p2_falls INTEGER,
            p3_falls INTEGER,
            p4_falls INTEGER,
            p1_damage INTEGER,
            p2_damage INTEGER,
            p3_damage INTEGER,
            p4_damage INTEGER,
            win TEXT NOT NULL,
            opponent TEXT
        )
    """)

    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_datetime ON games(datetime)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_win ON games(win)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_opponent ON games(opponent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_p1_character ON games(p1_character)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_p2_character ON games(p2_character)")

    # Create matchups table for aggregated matchup data
    # Groups by 4 characters only (opponent name comes from first game)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matchups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_game_date TEXT NOT NULL,
            last_game_date TEXT NOT NULL,
            p1_character TEXT,
            p2_character TEXT,
            p3_character TEXT,
            p4_character TEXT,
            opponent TEXT,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            win_loss_order TEXT,
            matchup_result REAL,
            UNIQUE (p1_character, p2_character, p3_character, p4_character)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matchups_opponent ON matchups(opponent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matchups_result ON matchups(matchup_result)")

    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


def save_game_result(result: dict) -> int:
    """
    Save a game result to the database and update the matchups table.

    Args:
        result: Dictionary with game data (same format as CSV row)

    Returns:
        int: The ID of the inserted game
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Insert the game
    cursor.execute("""
        INSERT INTO games (
            datetime,
            p1_character, p2_character, p3_character, p4_character,
            p1_kos, p2_kos, p3_kos, p4_kos,
            p1_falls, p2_falls, p3_falls, p4_falls,
            p1_damage, p2_damage, p3_damage, p4_damage,
            win, opponent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result['datetime'],
        result.get('p1_character', ''),
        result.get('p2_character', ''),
        result.get('p3_character', ''),
        result.get('p4_character', ''),
        result.get('p1_kos'),
        result.get('p2_kos'),
        result.get('p3_kos'),
        result.get('p4_kos'),
        result.get('p1_falls'),
        result.get('p2_falls'),
        result.get('p3_falls'),
        result.get('p4_falls'),
        result.get('p1_damage'),
        result.get('p2_damage'),
        result.get('p3_damage'),
        result.get('p4_damage'),
        result['win'],
        result.get('opponent', '')
    ))

    game_id = cursor.lastrowid

    # Update matchups incrementally
    _update_matchup_for_game(cursor, result)

    conn.commit()
    conn.close()

    return game_id


def _update_matchup_for_game(cursor, result: dict):
    """
    Incrementally update the matchups table for a single game.
    Groups by 4 characters only (opponent name comes from first game).
    Called internally by save_game_result().
    """
    p1 = result.get('p1_character', '')
    p2 = result.get('p2_character', '')
    p3 = result.get('p3_character', '')
    p4 = result.get('p4_character', '')
    opponent = result.get('opponent', '')
    game_date = result['datetime']
    is_win = result['win'] == 'Yes'

    # Check if matchup exists (grouped by 4 characters only)
    cursor.execute("""
        SELECT id, wins, losses, win_loss_order, first_game_date
        FROM matchups
        WHERE p1_character = ? AND p2_character = ? AND p3_character = ?
              AND p4_character = ?
    """, (p1, p2, p3, p4))

    existing = cursor.fetchone()

    if existing:
        # Update existing matchup (keep original opponent name)
        matchup_id = existing['id']
        wins = existing['wins'] + (1 if is_win else 0)
        losses = existing['losses'] + (0 if is_win else 1)
        win_loss_order = existing['win_loss_order'] + ('y' if is_win else 'n')

        # Calculate new matchup result
        if wins > losses:
            matchup_result = 1.0
        elif wins < losses:
            matchup_result = 0.0
        else:
            matchup_result = 0.5

        cursor.execute("""
            UPDATE matchups
            SET wins = ?, losses = ?, win_loss_order = ?,
                last_game_date = ?, matchup_result = ?
            WHERE id = ?
        """, (wins, losses, win_loss_order, game_date, matchup_result, matchup_id))
    else:
        # Create new matchup (use opponent name from this first game)
        wins = 1 if is_win else 0
        losses = 0 if is_win else 1
        win_loss_order = 'y' if is_win else 'n'
        matchup_result = 1.0 if is_win else 0.0

        cursor.execute("""
            INSERT INTO matchups (
                first_game_date, last_game_date,
                p1_character, p2_character, p3_character, p4_character,
                opponent, wins, losses, win_loss_order, matchup_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_date, game_date, p1, p2, p3, p4, opponent,
              wins, losses, win_loss_order, matchup_result))


def get_all_games():
    """
    Get all games.

    Returns:
        list: List of game dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM games ORDER BY datetime DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_game_count():
    """Get total number of games in database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def rebuild_matchups():
    """
    Rebuild the matchups table from games data.
    Groups games by all 4 characters only (opponent name from first game).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Clear existing matchups
    cursor.execute("DELETE FROM matchups")

    # Get all games ordered by datetime for correct win_loss_order
    cursor.execute("""
        SELECT datetime, p1_character, p2_character, p3_character, p4_character,
               opponent, win
        FROM games
        ORDER BY datetime ASC
    """)
    games = cursor.fetchall()

    # Group games by matchup key (4 characters only, not opponent)
    matchups = {}
    for game in games:
        key = (
            game['p1_character'],
            game['p2_character'],
            game['p3_character'],
            game['p4_character']
        )

        if key not in matchups:
            matchups[key] = {
                'first_game_date': game['datetime'],
                'last_game_date': game['datetime'],
                'p1_character': game['p1_character'],
                'p2_character': game['p2_character'],
                'p3_character': game['p3_character'],
                'p4_character': game['p4_character'],
                'opponent': game['opponent'],  # Use opponent from first game
                'wins': 0,
                'losses': 0,
                'win_loss_order': ''
            }

        matchup = matchups[key]
        matchup['last_game_date'] = game['datetime']

        if game['win'] == 'Yes':
            matchup['wins'] += 1
            matchup['win_loss_order'] += 'y'
        else:
            matchup['losses'] += 1
            matchup['win_loss_order'] += 'n'

    # Calculate matchup_result and insert into table
    for matchup in matchups.values():
        wins = matchup['wins']
        losses = matchup['losses']

        if wins > losses:
            matchup_result = 1.0
        elif wins < losses:
            matchup_result = 0.0
        else:
            matchup_result = 0.5

        cursor.execute("""
            INSERT INTO matchups (
                first_game_date, last_game_date,
                p1_character, p2_character, p3_character, p4_character,
                opponent, wins, losses, win_loss_order, matchup_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            matchup['first_game_date'],
            matchup['last_game_date'],
            matchup['p1_character'],
            matchup['p2_character'],
            matchup['p3_character'],
            matchup['p4_character'],
            matchup['opponent'],
            matchup['wins'],
            matchup['losses'],
            matchup['win_loss_order'],
            matchup_result
        ))

    conn.commit()
    conn.close()

    print(f"Rebuilt {len(matchups)} matchups from {len(games)} games")
    return len(matchups)


def get_all_matchups():
    """
    Get all matchups.

    Returns:
        list: List of matchup dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM matchups ORDER BY last_game_date DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_matchup_count():
    """Get total number of matchups in database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM matchups")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_today_stats():
    """Get today's statistics."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    # Get today's games
    cursor.execute("""
        SELECT win FROM games WHERE datetime LIKE ?
    """, (f"{today}%",))
    today_games = cursor.fetchall()

    # Get unique matchups played today
    cursor.execute("""
        SELECT DISTINCT p1_character, p2_character, p3_character, p4_character
        FROM games WHERE datetime LIKE ?
    """, (f"{today}%",))
    today_matchups = cursor.fetchall()

    conn.close()

    total_today = len(today_games)
    wins_today = sum(1 for g in today_games if g['win'] == 'Yes')

    # Get matchup win % for matchups played today
    matchups = get_all_matchups()
    today_matchup_keys = set(
        (m['p1_character'], m['p2_character'], m['p3_character'], m['p4_character'])
        for m in today_matchups
    )

    matching_matchups = [
        m for m in matchups
        if (m['p1_character'], m['p2_character'], m['p3_character'], m['p4_character']) in today_matchup_keys
    ]

    if matching_matchups:
        matchup_wins = sum(1 for m in matching_matchups if m['matchup_result'] == 1.0)
        matchup_losses = sum(1 for m in matching_matchups if m['matchup_result'] == 0.0)
        matchup_ties = len(matching_matchups) - matchup_wins - matchup_losses
        matchup_win_pct = matchup_wins / len(matching_matchups) * 100
    else:
        matchup_wins = 0
        matchup_losses = 0
        matchup_ties = 0
        matchup_win_pct = 0

    return {
        "today_games": total_today,
        "today_wins": wins_today,
        "today_losses": total_today - wins_today,
        "today_matchups": len(today_matchups),
        "matchup_wins": matchup_wins,
        "matchup_losses": matchup_losses,
        "matchup_ties": matchup_ties,
        "matchup_win_pct": round(matchup_win_pct, 1)
    }


def get_last_month_stats():
    """Get last month's matchup statistics."""
    from datetime import datetime, timedelta

    # Get date 30 days ago
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    # Get unique matchups from games in last month
    cursor.execute("""
        SELECT DISTINCT p1_character, p2_character, p3_character, p4_character
        FROM games WHERE datetime >= ?
    """, (month_ago,))
    month_matchup_keys = cursor.fetchall()

    conn.close()

    # Get current matchup results for those matchups
    matchups = get_all_matchups()
    month_keys = set(
        (m['p1_character'], m['p2_character'], m['p3_character'], m['p4_character'])
        for m in month_matchup_keys
    )

    matching_matchups = [
        m for m in matchups
        if (m['p1_character'], m['p2_character'], m['p3_character'], m['p4_character']) in month_keys
    ]

    if matching_matchups:
        matchup_wins = sum(1 for m in matching_matchups if m['matchup_result'] == 1.0)
        matchup_losses = sum(1 for m in matching_matchups if m['matchup_result'] == 0.0)
        matchup_ties = len(matching_matchups) - matchup_wins - matchup_losses
        total = len(matching_matchups)
        matchup_win_pct = matchup_wins / total * 100 if total > 0 else 0
    else:
        matchup_wins = 0
        matchup_losses = 0
        matchup_ties = 0
        matchup_win_pct = 0

    return {
        "month_matchups": len(matching_matchups),
        "matchup_wins": matchup_wins,
        "matchup_losses": matchup_losses,
        "matchup_ties": matchup_ties,
        "matchup_win_pct": round(matchup_win_pct, 1)
    }


def get_game_by_id(game_id: int):
    """Get a single game by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_game(game_id: int, data: dict):
    """
    Update a game and rebuild matchups.

    Args:
        game_id: ID of game to update
        data: Dictionary with fields to update

    Returns:
        bool: True if successful
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Build update query dynamically
    fields = []
    values = []
    for key in ['datetime', 'p1_character', 'p2_character', 'p3_character', 'p4_character',
                'p1_kos', 'p2_kos', 'p3_kos', 'p4_kos',
                'p1_falls', 'p2_falls', 'p3_falls', 'p4_falls',
                'p1_damage', 'p2_damage', 'p3_damage', 'p4_damage',
                'win', 'opponent']:
        if key in data:
            fields.append(f"{key} = ?")
            values.append(data[key])

    if not fields:
        conn.close()
        return False

    values.append(game_id)
    cursor.execute(f"UPDATE games SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()
    conn.close()

    # Rebuild matchups since game data changed
    rebuild_matchups()
    return True


def delete_game(game_id: int):
    """
    Delete a game and rebuild matchups.

    Args:
        game_id: ID of game to delete

    Returns:
        bool: True if successful
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM games WHERE id = ?", (game_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()

    if deleted:
        # Rebuild matchups since game was removed
        rebuild_matchups()

    return deleted


def migrate_to_nullable_stats():
    """
    Migrate existing database to allow NULL values for all player stats.
    This removes DEFAULT 0 constraints from KO/falls/damage columns.
    """
    conn = get_connection()
    cursor = conn.cursor()

    print("Migrating database schema to allow NULL stats...")

    # Create new table with nullable stats
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            p1_character TEXT,
            p2_character TEXT,
            p3_character TEXT,
            p4_character TEXT,
            p1_kos INTEGER,
            p2_kos INTEGER,
            p3_kos INTEGER,
            p4_kos INTEGER,
            p1_falls INTEGER,
            p2_falls INTEGER,
            p3_falls INTEGER,
            p4_falls INTEGER,
            p1_damage INTEGER,
            p2_damage INTEGER,
            p3_damage INTEGER,
            p4_damage INTEGER,
            win TEXT NOT NULL,
            opponent TEXT
        )
    """)

    # Copy data from old table to new table
    cursor.execute("""
        INSERT INTO games_new SELECT * FROM games
    """)

    # Drop old table
    cursor.execute("DROP TABLE games")

    # Rename new table to games
    cursor.execute("ALTER TABLE games_new RENAME TO games")

    # Recreate indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_datetime ON games(datetime)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_win ON games(win)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_opponent ON games(opponent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_p1_character ON games(p1_character)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_p2_character ON games(p2_character)")

    conn.commit()
    conn.close()

    print("Migration complete! All player stats can now be NULL.")


if __name__ == "__main__":
    # Initialize database and rebuild matchups when run directly
    init_db()
    print(f"Game count: {get_game_count()}")
    rebuild_matchups()
    print(f"Matchup count: {get_matchup_count()}")
