#!/usr/bin/env python3
"""
FastAPI backend for SmashStats web dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from database import (
    init_db, get_all_games, get_all_matchups, get_game_count, get_matchup_count,
    get_today_stats, get_last_month_stats, get_game_by_id, save_game_result, update_game, delete_game,
    get_character_stats
)

# Initialize database on startup
init_db()

app = FastAPI(title="SmashStats", description="Super Smash Bros match tracking")

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class GameCreate(BaseModel):
    datetime: str
    p1_character: str = ""
    p2_character: str = ""
    p3_character: str = ""
    p4_character: str = ""
    p1_kos: Optional[int] = None
    p2_kos: Optional[int] = None
    p3_kos: Optional[int] = None
    p4_kos: Optional[int] = None
    p1_damage: Optional[int] = None
    p2_damage: Optional[int] = None
    p3_damage: Optional[int] = None
    p4_damage: Optional[int] = None
    win: str
    opponent: str = ""


class GameUpdate(BaseModel):
    datetime: Optional[str] = None
    p1_character: Optional[str] = None
    p2_character: Optional[str] = None
    p3_character: Optional[str] = None
    p4_character: Optional[str] = None
    p1_kos: Optional[int] = None
    p2_kos: Optional[int] = None
    p3_kos: Optional[int] = None
    p4_kos: Optional[int] = None
    p1_damage: Optional[int] = None
    p2_damage: Optional[int] = None
    p3_damage: Optional[int] = None
    p4_damage: Optional[int] = None
    win: Optional[str] = None
    opponent: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main dashboard page."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text()


@app.get("/api/games")
async def api_games():
    """Get all games."""
    games = get_all_games()
    return {
        "count": len(games),
        "games": games
    }


@app.get("/api/games/{game_id}")
async def api_get_game(game_id: int):
    """Get a single game by ID."""
    game = get_game_by_id(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@app.post("/api/games")
async def api_create_game(game: GameCreate):
    """Create a new game."""
    game_id = save_game_result(game.model_dump())
    return {"id": game_id, "message": "Game created"}


@app.put("/api/games/{game_id}")
async def api_update_game(game_id: int, game: GameUpdate):
    """Update an existing game."""
    existing = get_game_by_id(game_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Game not found")

    # Only include fields that were explicitly provided (exclude unset fields)
    # This allows None values to be saved (clearing a field)
    update_data = game.model_dump(exclude_unset=True)
    if update_data:
        update_game(game_id, update_data)

    return {"message": "Game updated"}


@app.delete("/api/games/{game_id}")
async def api_delete_game(game_id: int):
    """Delete a game."""
    if not delete_game(game_id):
        raise HTTPException(status_code=404, detail="Game not found")
    return {"message": "Game deleted"}


@app.get("/api/matchups")
async def api_matchups():
    """Get all matchups."""
    matchups = get_all_matchups()
    return {
        "count": len(matchups),
        "matchups": matchups
    }


@app.get("/api/stats")
async def api_stats():
    """Get summary statistics."""
    games = get_all_games()
    total = len(games)
    wins = sum(1 for g in games if g['win'] == 'Yes')
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0

    return {
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_matchups": get_matchup_count()
    }


@app.get("/api/stats/today")
async def api_today_stats():
    """Get today's statistics."""
    return get_today_stats()


@app.get("/api/stats/month")
async def api_month_stats():
    """Get last month's statistics."""
    return get_last_month_stats()


@app.get("/api/stats/overall")
async def api_overall_stats():
    """Get overall matchup statistics for charts."""
    matchups = get_all_matchups()
    total = len(matchups)
    wins = sum(1 for m in matchups if m['matchup_result'] == 1.0)
    losses = sum(1 for m in matchups if m['matchup_result'] == 0.0)
    ties = total - wins - losses
    # Count ties as 0.5 wins and 0.5 losses for display
    adjusted_wins = wins + (0.5 * ties)
    adjusted_losses = losses + (0.5 * ties)
    win_pct = (adjusted_wins / total * 100) if total > 0 else 0

    return {
        "total_matchups": total,
        "matchup_wins": adjusted_wins,
        "matchup_losses": adjusted_losses,
        "matchup_ties": 0,  # Don't show ties separately
        "matchup_win_pct": round(win_pct, 1)
    }


@app.get("/api/stats/characters")
async def api_character_stats():
    """Get character statistics for p1 and p2."""
    return get_character_stats()


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("SMASHSTATS WEB DASHBOARD")
    print("=" * 50)
    print("\nAccess from this computer: http://localhost:8000")
    print("Access from phone (same network): http://<your-ip>:8000")
    print("\nPress Ctrl+C to stop.\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
