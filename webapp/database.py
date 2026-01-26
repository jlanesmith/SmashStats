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
    # A matchup is a consecutive series of games with the same 4 characters
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
            matchup_result REAL
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
            p1_damage, p2_damage, p3_damage, p4_damage,
            win, opponent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    A matchup is a consecutive series of games with the same 4 characters.
    Called internally by save_game_result().

    Note: For simplicity, new games always create a new matchup or extend
    the most recent one if characters match. Full rebuild handles historical data.
    """
    p1 = result.get('p1_character', '')
    p2 = result.get('p2_character', '')
    p3 = result.get('p3_character', '')
    p4 = result.get('p4_character', '')
    opponent = result.get('opponent', '')
    game_date = result['datetime']
    is_win = result['win'] == 'Yes'

    # Check if the most recent matchup has the same 4 characters
    cursor.execute("""
        SELECT id, wins, losses, win_loss_order, first_game_date,
               p1_character, p2_character, p3_character, p4_character
        FROM matchups
        ORDER BY last_game_date DESC
        LIMIT 1
    """)

    existing = cursor.fetchone()

    # Only extend if it's the same 4 characters
    if existing and (existing['p1_character'] == p1 and
                     existing['p2_character'] == p2 and
                     existing['p3_character'] == p3 and
                     existing['p4_character'] == p4):
        # Update existing matchup
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
        # Create new matchup
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
    A matchup is a consecutive series of games with the same 4 characters.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Drop and recreate matchups table (to remove old UNIQUE constraint if present)
    cursor.execute("DROP TABLE IF EXISTS matchups")
    cursor.execute("""
        CREATE TABLE matchups (
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
            matchup_result REAL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matchups_opponent ON matchups(opponent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matchups_result ON matchups(matchup_result)")

    # Get all games ordered by datetime for correct consecutive grouping
    cursor.execute("""
        SELECT datetime, p1_character, p2_character, p3_character, p4_character,
               opponent, win
        FROM games
        ORDER BY datetime ASC
    """)
    games = cursor.fetchall()

    # Build matchups from consecutive games with same 4 characters
    matchups = []
    current_matchup = None

    for game in games:
        game_chars = (
            game['p1_character'],
            game['p2_character'],
            game['p3_character'],
            game['p4_character']
        )

        # Check if this game continues the current matchup
        if current_matchup is not None:
            current_chars = (
                current_matchup['p1_character'],
                current_matchup['p2_character'],
                current_matchup['p3_character'],
                current_matchup['p4_character']
            )

            if game_chars == current_chars:
                # Extend current matchup
                current_matchup['last_game_date'] = game['datetime']
                if game['win'] == 'Yes':
                    current_matchup['wins'] += 1
                    current_matchup['win_loss_order'] += 'y'
                else:
                    current_matchup['losses'] += 1
                    current_matchup['win_loss_order'] += 'n'
                continue

        # Start a new matchup (either first game or characters changed)
        if current_matchup is not None:
            matchups.append(current_matchup)

        current_matchup = {
            'first_game_date': game['datetime'],
            'last_game_date': game['datetime'],
            'p1_character': game['p1_character'],
            'p2_character': game['p2_character'],
            'p3_character': game['p3_character'],
            'p4_character': game['p4_character'],
            'opponent': game['opponent'],
            'wins': 1 if game['win'] == 'Yes' else 0,
            'losses': 0 if game['win'] == 'Yes' else 1,
            'win_loss_order': 'y' if game['win'] == 'Yes' else 'n'
        }

    # Don't forget the last matchup
    if current_matchup is not None:
        matchups.append(current_matchup)

    # Calculate matchup_result and insert into table
    for matchup in matchups:
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
        # Count ties as 0.5 wins and 0.5 losses for display
        adjusted_wins = matchup_wins + (0.5 * matchup_ties)
        adjusted_losses = matchup_losses + (0.5 * matchup_ties)
        matchup_win_pct = (adjusted_wins / len(matching_matchups)) * 100
    else:
        adjusted_wins = 0
        adjusted_losses = 0
        matchup_win_pct = 0

    return {
        "today_games": total_today,
        "today_wins": wins_today,
        "today_losses": total_today - wins_today,
        "today_matchups": len(today_matchups),
        "matchup_wins": adjusted_wins,
        "matchup_losses": adjusted_losses,
        "matchup_ties": 0,  # Don't show ties separately
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
        # Count ties as 0.5 wins and 0.5 losses for display
        adjusted_wins = matchup_wins + (0.5 * matchup_ties)
        adjusted_losses = matchup_losses + (0.5 * matchup_ties)
        total = len(matching_matchups)
        matchup_win_pct = (adjusted_wins / total) * 100 if total > 0 else 0
    else:
        adjusted_wins = 0
        adjusted_losses = 0
        matchup_win_pct = 0

    return {
        "month_matchups": len(matching_matchups),
        "matchup_wins": adjusted_wins,
        "matchup_losses": adjusted_losses,
        "matchup_ties": 0,  # Don't show ties separately
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


def get_character_stats():
    """
    Get character statistics for p1 and p2 based on matchups.

    Returns:
        dict: Stats for each player position with character counts and success rates
    """
    conn = get_connection()
    cursor = conn.cursor()

    result = {'p1': [], 'p2': []}

    for player in ['p1', 'p2']:
        cursor.execute(f"""
            SELECT
                {player}_character as character,
                COUNT(*) as count,
                SUM(CASE WHEN matchup_result = 1.0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN matchup_result = 0.5 THEN 1 ELSE 0 END) as ties,
                SUM(CASE WHEN matchup_result = 0.0 THEN 1 ELSE 0 END) as losses,
                AVG(matchup_result) * 100 as success_rate
            FROM matchups
            WHERE {player}_character IS NOT NULL AND {player}_character != ''
            GROUP BY {player}_character
            ORDER BY count DESC
        """)

        for row in cursor.fetchall():
            result[player].append({
                'character': row['character'],
                'count': row['count'],
                'wins': row['wins'],
                'ties': row['ties'],
                'losses': row['losses'],
                'win_pct': round(row['success_rate'], 1)
            })

    conn.close()
    return result


def get_character_stats_month():
    """
    Get character statistics for p1 and p2 based on matchups from the last 30 days.

    Returns:
        dict: Stats for each player position with character counts and success rates
    """
    conn = get_connection()
    cursor = conn.cursor()

    result = {'p1': [], 'p2': []}

    for player in ['p1', 'p2']:
        cursor.execute(f"""
            SELECT
                {player}_character as character,
                COUNT(*) as count,
                SUM(CASE WHEN matchup_result = 1.0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN matchup_result = 0.5 THEN 1 ELSE 0 END) as ties,
                SUM(CASE WHEN matchup_result = 0.0 THEN 1 ELSE 0 END) as losses,
                AVG(matchup_result) * 100 as success_rate
            FROM matchups
            WHERE {player}_character IS NOT NULL
                AND {player}_character != ''
                AND last_game_date >= datetime('now', '-30 days')
            GROUP BY {player}_character
            ORDER BY count DESC
        """)

        for row in cursor.fetchall():
            result[player].append({
                'character': row['character'],
                'count': row['count'],
                'wins': row['wins'],
                'ties': row['ties'],
                'losses': row['losses'],
                'win_pct': round(row['success_rate'], 1)
            })

    conn.close()
    return result


def get_character_ko_damage_stats():
    """
    Get average KOs and damage for p1 and p2 characters.

    Returns:
        dict: Stats for each player position with avg KOs and damage per character
    """
    conn = get_connection()
    cursor = conn.cursor()

    result = {'p1': [], 'p2': []}

    for player in ['p1', 'p2']:
        cursor.execute(f"""
            SELECT
                {player}_character as character,
                COUNT(*) as count,
                AVG({player}_kos) as avg_kos,
                AVG({player}_damage) as avg_damage
            FROM games
            WHERE {player}_character IS NOT NULL AND {player}_character != ''
                AND {player}_kos IS NOT NULL AND {player}_damage IS NOT NULL
            GROUP BY {player}_character
            ORDER BY count DESC
        """)

        for row in cursor.fetchall():
            result[player].append({
                'character': row['character'],
                'count': row['count'],
                'avg_kos': round(row['avg_kos'], 1) if row['avg_kos'] else 0,
                'avg_damage': round(row['avg_damage'], 0) if row['avg_damage'] else 0
            })

    conn.close()
    return result


def get_opponent_ko_damage_stats():
    """
    Get average KOs and damage for opponent characters (p3 and p4 combined).

    Returns:
        list: Stats for opponent characters with avg KOs and damage
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Combine p3 and p4 character data using UNION ALL
    cursor.execute("""
        SELECT
            character,
            COUNT(*) as count,
            AVG(kos) as avg_kos,
            AVG(damage) as avg_damage
        FROM (
            SELECT p3_character as character, p3_kos as kos, p3_damage as damage
            FROM games
            WHERE p3_character IS NOT NULL AND p3_character != ''
                AND p3_kos IS NOT NULL AND p3_damage IS NOT NULL
            UNION ALL
            SELECT p4_character as character, p4_kos as kos, p4_damage as damage
            FROM games
            WHERE p4_character IS NOT NULL AND p4_character != ''
                AND p4_kos IS NOT NULL AND p4_damage IS NOT NULL
        )
        GROUP BY character
        ORDER BY count DESC
    """)

    result = []
    for row in cursor.fetchall():
        result.append({
            'character': row['character'],
            'count': row['count'],
            'avg_kos': round(row['avg_kos'], 1) if row['avg_kos'] else 0,
            'avg_damage': round(row['avg_damage'], 0) if row['avg_damage'] else 0
        })

    conn.close()
    return result


def get_opponent_character_stats():
    """
    Get character statistics for opponents (p3 and p4 combined) based on matchups.

    Returns:
        dict: Stats for opponent characters with counts and success rates
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Combine p3 and p4 character data using UNION ALL
    cursor.execute("""
        SELECT
            character,
            COUNT(*) as count,
            SUM(CASE WHEN matchup_result = 1.0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN matchup_result = 0.5 THEN 1 ELSE 0 END) as ties,
            SUM(CASE WHEN matchup_result = 0.0 THEN 1 ELSE 0 END) as losses,
            AVG(matchup_result) * 100 as success_rate
        FROM (
            SELECT p3_character as character, matchup_result
            FROM matchups
            WHERE p3_character IS NOT NULL AND p3_character != ''
            UNION ALL
            SELECT p4_character as character, matchup_result
            FROM matchups
            WHERE p4_character IS NOT NULL AND p4_character != ''
        )
        GROUP BY character
        ORDER BY count DESC
    """)

    result = []
    for row in cursor.fetchall():
        result.append({
            'character': row['character'],
            'count': row['count'],
            'wins': row['wins'],
            'ties': row['ties'],
            'losses': row['losses'],
            'win_pct': round(row['success_rate'], 1)
        })

    conn.close()
    return result


def get_weekday_stats():
    """
    Get performance statistics grouped by day of week.

    Returns:
        list: Stats for each day of week (Sunday=0 through Saturday=6)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # SQLite strftime %w returns 0=Sunday, 1=Monday, ..., 6=Saturday
    cursor.execute("""
        SELECT
            CAST(strftime('%w', last_game_date) AS INTEGER) as weekday,
            COUNT(*) as count,
            AVG(matchup_result) * 100 as win_pct
        FROM matchups
        GROUP BY weekday
        ORDER BY weekday
    """)

    # Initialize all days with zero values
    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    result = [{'day': day, 'count': 0, 'win_pct': 0} for day in days]

    for row in cursor.fetchall():
        weekday = row['weekday']
        result[weekday] = {
            'day': days[weekday],
            'count': row['count'],
            'win_pct': round(row['win_pct'], 1) if row['win_pct'] else 0
        }

    conn.close()
    return result


def get_halfhour_stats():
    """
    Get performance statistics grouped by 15-minute time slot from 4pm to 1am.

    Returns:
        list: Stats for each 15-minute slot that has data
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Extract hour and determine which 15-minute quarter of the hour
    # Filter to only include 4pm (16:00) to 1am (01:00)
    cursor.execute("""
        SELECT
            CAST(strftime('%H', last_game_date) AS INTEGER) as hour,
            CAST(strftime('%M', last_game_date) AS INTEGER) / 15 as quarter,
            COUNT(*) as count,
            AVG(matchup_result) * 100 as success_rate
        FROM matchups
        WHERE CAST(strftime('%H', last_game_date) AS INTEGER) >= 16
           OR CAST(strftime('%H', last_game_date) AS INTEGER) < 2
        GROUP BY hour, quarter
        ORDER BY
            CASE WHEN hour >= 16 THEN hour - 16 ELSE hour + 8 END,
            quarter
    """)

    result = []
    for row in cursor.fetchall():
        hour = row['hour']
        quarter = row['quarter']
        minute = quarter * 15

        # Convert to 12-hour format with AM/PM
        if hour == 0:
            display_hour = 12
            period = 'AM'
        elif hour < 12:
            display_hour = hour
            period = 'AM'
        elif hour == 12:
            display_hour = 12
            period = 'PM'
        else:
            display_hour = hour - 12
            period = 'PM'

        time_label = f"{display_hour}:{minute:02d} {period}"

        result.append({
            'time': time_label,
            'sort_key': (hour - 16) % 24 * 4 + quarter,  # For sorting
            'count': row['count'],
            'success_rate': round(row['success_rate'], 0) if row['success_rate'] else 0
        })

    conn.close()
    return result


def get_daily_success_rate():
    """
    Get success rate for each day that has matchups.

    Returns:
        list: Daily success rates with date and percentage
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            DATE(last_game_date) as date,
            AVG(matchup_result) * 100 as success_rate
        FROM matchups
        GROUP BY DATE(last_game_date)
        ORDER BY date ASC
    """)

    result = []
    for row in cursor.fetchall():
        result.append({
            'date': row['date'],
            'success_rate': round(row['success_rate'], 0) if row['success_rate'] else 0
        })

    conn.close()
    return result


def get_order_stats():
    """
    Get counts of each win_loss_order pattern, sorted by length then count.

    Returns:
        list: Order patterns with their counts
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            win_loss_order as pattern,
            COUNT(*) as count
        FROM matchups
        WHERE win_loss_order IS NOT NULL AND win_loss_order != ''
        GROUP BY win_loss_order
        ORDER BY LENGTH(win_loss_order) ASC, count DESC
    """)

    result = []
    for row in cursor.fetchall():
        result.append({
            'pattern': row['pattern'],
            'count': row['count']
        })

    conn.close()
    return result


def get_opponent_pairs_stats():
    """
    Get stats for each opponent character pairing (p3 & p4).
    Combines A & B with B & A by sorting characters alphabetically.

    Returns:
        list: Opponent pairs with count and success rate, sorted by count desc
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Use MIN and MAX to normalize the pair order (alphabetically)
    cursor.execute("""
        SELECT
            MIN(p3_character, p4_character) || ' & ' || MAX(p3_character, p4_character) as pair,
            COUNT(*) as count,
            AVG(matchup_result) * 100 as success_rate
        FROM matchups
        WHERE p3_character IS NOT NULL AND p3_character != ''
          AND p4_character IS NOT NULL AND p4_character != ''
        GROUP BY MIN(p3_character, p4_character), MAX(p3_character, p4_character)
        HAVING count >= 3
        ORDER BY count DESC
    """)

    result = []
    for row in cursor.fetchall():
        result.append({
            'pair': row['pair'],
            'count': row['count'],
            'success_rate': round(row['success_rate'], 0) if row['success_rate'] else 0
        })

    conn.close()
    return result


def get_streak_stats():
    """
    Calculate the maximum win streak and loss streak from matchups.

    Returns:
        dict: max_win_streak and max_loss_streak
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get all matchups ordered by date
    cursor.execute("""
        SELECT matchup_result
        FROM matchups
        ORDER BY last_game_date ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for row in rows:
        result = row['matchup_result']

        if result == 1.0:  # Win
            current_win_streak += 1
            current_loss_streak = 0
            if current_win_streak > max_win_streak:
                max_win_streak = current_win_streak
        elif result == 0.0:  # Loss
            current_loss_streak += 1
            current_win_streak = 0
            if current_loss_streak > max_loss_streak:
                max_loss_streak = current_loss_streak
        else:  # Tie (0.5) - breaks both streaks
            current_win_streak = 0
            current_loss_streak = 0

    return {
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak
    }


def migrate_to_nullable_stats():
    """
    Migrate existing database to allow NULL values for all player stats
    and remove falls columns (no longer tracked).
    """
    conn = get_connection()
    cursor = conn.cursor()

    print("Migrating database schema...")
    print("- Allowing NULL stats")
    print("- Removing falls columns")

    # Create new table without falls columns
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
            p1_damage INTEGER,
            p2_damage INTEGER,
            p3_damage INTEGER,
            p4_damage INTEGER,
            win TEXT NOT NULL,
            opponent TEXT
        )
    """)

    # Copy data from old table to new table (excluding falls columns)
    cursor.execute("""
        INSERT INTO games_new (
            id, datetime,
            p1_character, p2_character, p3_character, p4_character,
            p1_kos, p2_kos, p3_kos, p4_kos,
            p1_damage, p2_damage, p3_damage, p4_damage,
            win, opponent
        )
        SELECT
            id, datetime,
            p1_character, p2_character, p3_character, p4_character,
            p1_kos, p2_kos, p3_kos, p4_kos,
            p1_damage, p2_damage, p3_damage, p4_damage,
            win, opponent
        FROM games
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

    print("Migration complete!")
    print("- All player stats can now be NULL")
    print("- Falls columns removed")


if __name__ == "__main__":
    # Initialize database and rebuild matchups when run directly
    init_db()
    print(f"Game count: {get_game_count()}")
    rebuild_matchups()
    print(f"Matchup count: {get_matchup_count()}")
