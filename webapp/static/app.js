// SmashStats Frontend JavaScript

// Undo system - stores last action
let lastAction = null;

// Chart instances
let todayChart = null;
let overallChart = null;
let monthChart = null;

// Tab switching
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('tab-active');
        el.classList.add('text-gray-500');
    });

    document.getElementById(`content-${tabName}`).classList.remove('hidden');
    const activeTab = document.getElementById(`tab-${tabName}`);
    activeTab.classList.add('tab-active');
    activeTab.classList.remove('text-gray-500');
}

// Format date for display
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Format stat value - show "N/A" for null/undefined
function formatStat(value) {
    return (value === null || value === undefined) ? 'N/A' : value;
}

// Safe add - treats null as 0 for calculations
function safeAdd(...values) {
    return values.reduce((sum, val) => sum + (val || 0), 0);
}

// Format team stat - shows N/A if all values are null
function formatTeamStat(val1, val2) {
    if (val1 === null && val2 === null) {
        return 'N/A';
    }
    return safeAdd(val1, val2);
}

// Format date for input field (keep as local time, no timezone conversion)
function formatDateForInput(dateStr) {
    // Database format: "2024-01-24 14:00:00" -> Input format: "2024-01-24T14:00"
    return dateStr.replace(' ', 'T').slice(0, 16);
}

// Format date from input to database format (keep as entered, no timezone conversion)
function formatDateFromInput(inputValue) {
    // Input format: "2024-01-24T14:00" -> Database format: "2024-01-24 14:00:00"
    return inputValue.replace('T', ' ') + ':00';
}

// Format win/loss order as colored boxes
function formatWinLossOrder(order) {
    return order.split('').map(c => {
        if (c === 'y') {
            return '<span class="inline-block w-4 h-4 bg-green-500 rounded mr-0.5"></span>';
        } else {
            return '<span class="inline-block w-4 h-4 bg-red-500 rounded mr-0.5"></span>';
        }
    }).join('');
}

// Format matchup result
function formatMatchupResult(result) {
    if (result === 1.0) {
        return '<span class="win">W</span>';
    } else if (result === 0.0) {
        return '<span class="loss">L</span>';
    } else {
        return '<span class="text-yellow-600 font-semibold">T</span>';
    }
}

// Get row class based on result
function getResultRowClass(result) {
    if (result === 1.0) return 'result-1';
    if (result === 0.0) return 'result-0';
    return 'result-half';
}

// Load and display games
async function loadGames() {
    try {
        const response = await fetch('/api/games');
        const data = await response.json();

        const tbody = document.getElementById('games-table');
        tbody.innerHTML = data.games.map(game => {
            const team1KOs = formatTeamStat(game.p1_kos, game.p2_kos);
            const team2KOs = formatTeamStat(game.p3_kos, game.p4_kos);
            const team1Falls = formatTeamStat(game.p1_falls, game.p2_falls);
            const team2Falls = formatTeamStat(game.p3_falls, game.p4_falls);
            const teamKOs = `${team1KOs} - ${team2KOs}`;
            const teamFalls = `${team1Falls} - ${team2Falls}`;

            return `
                <tr class="${game.win === 'Yes' ? 'result-1' : 'result-0'}">
                    <td class="px-3 py-2 whitespace-nowrap">${formatDate(game.datetime)}</td>
                    <td class="px-3 py-2">${game.p1_character}</td>
                    <td class="px-3 py-2">${game.p2_character}</td>
                    <td class="px-3 py-2">${game.p3_character}</td>
                    <td class="px-3 py-2">${game.p4_character}</td>
                    <td class="px-3 py-2 ${game.win === 'Yes' ? 'win' : 'loss'}">${game.win === 'Yes' ? 'Win' : 'Loss'}</td>
                    <td class="px-3 py-2">${game.opponent || '-'}</td>
                    <td class="px-3 py-2">${teamKOs}</td>
                    <td class="px-3 py-2">${teamFalls}</td>
                    <td class="px-3 py-2 whitespace-nowrap">
                        <button onclick="openEditModal(${game.id})" class="text-blue-500 hover:text-blue-700 mr-2">Edit</button>
                        <button onclick="openDeleteModal(${game.id})" class="text-red-500 hover:text-red-700">Delete</button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading games:', error);
    }
}

// Load and display matchups
async function loadMatchups() {
    try {
        const response = await fetch('/api/matchups');
        const data = await response.json();

        const tbody = document.getElementById('matchups-table');
        tbody.innerHTML = data.matchups.map(m => `
            <tr class="${getResultRowClass(m.matchup_result)}">
                <td class="px-3 py-2 font-medium">${m.opponent || '-'}</td>
                <td class="px-3 py-2">${m.p1_character}</td>
                <td class="px-3 py-2">${m.p2_character}</td>
                <td class="px-3 py-2">${m.p3_character}</td>
                <td class="px-3 py-2">${m.p4_character}</td>
                <td class="px-3 py-2 font-medium">${m.wins} - ${m.losses}</td>
                <td class="px-3 py-2">${formatWinLossOrder(m.win_loss_order)}</td>
                <td class="px-3 py-2">${formatMatchupResult(m.matchup_result)}</td>
                <td class="px-3 py-2 whitespace-nowrap">${formatDate(m.last_game_date)}</td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading matchups:', error);
    }
}

// Load today's stats
async function loadTodayStats() {
    try {
        const response = await fetch('/api/stats/today');
        const data = await response.json();

        document.getElementById('today-stats').innerHTML = `
            <span class="font-semibold">Today:</span>
            ${data.today_games} games |
            ${data.today_matchups} matchups |
            ${data.matchup_win_pct}% matchup win rate
        `;
    } catch (error) {
        console.error('Error loading today stats:', error);
    }
}

// Load overall stats
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        document.getElementById('overall-stats').innerHTML = `
            <span class="font-semibold">Overall:</span>
            ${data.total_games} games |
            ${data.wins}W - ${data.losses}L (${data.win_rate}%) |
            ${data.total_matchups} matchups
        `;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Create pie chart helper
function createPieChart(canvasId, wins, losses, ties) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const total = wins + losses + ties;

    // Handle empty data
    if (total === 0) {
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['No Data'],
                datasets: [{
                    data: [1],
                    backgroundColor: ['#e5e7eb']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    const data = {
        labels: ties > 0 ? ['Wins', 'Losses', 'Ties'] : ['Wins', 'Losses'],
        datasets: [{
            data: ties > 0 ? [wins, losses, ties] : [wins, losses],
            backgroundColor: ties > 0
                ? ['#22c55e', '#ef4444', '#eab308']
                : ['#22c55e', '#ef4444'],
            borderWidth: 2,
            borderColor: '#ffffff'
        }]
    };

    return new Chart(ctx, {
        type: 'pie',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 15 }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const pct = ((value / total) * 100).toFixed(1);
                            return `${context.label}: ${value} (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Load and render charts
async function loadCharts() {
    try {
        // Fetch all stats in parallel
        const [todayRes, overallRes, monthRes] = await Promise.all([
            fetch('/api/stats/today'),
            fetch('/api/stats/overall'),
            fetch('/api/stats/month')
        ]);

        const todayData = await todayRes.json();
        const overallData = await overallRes.json();
        const monthData = await monthRes.json();

        // Destroy existing charts if they exist
        if (todayChart) todayChart.destroy();
        if (overallChart) overallChart.destroy();
        if (monthChart) monthChart.destroy();

        // Create today's chart
        todayChart = createPieChart(
            'today-chart',
            todayData.matchup_wins,
            todayData.matchup_losses,
            todayData.matchup_ties
        );
        document.getElementById('today-chart-stats').innerHTML =
            `${todayData.today_matchups} matchups | ${todayData.matchup_win_pct}% win rate`;

        // Create overall chart
        overallChart = createPieChart(
            'overall-chart',
            overallData.matchup_wins,
            overallData.matchup_losses,
            overallData.matchup_ties
        );
        document.getElementById('overall-chart-stats').innerHTML =
            `${overallData.total_matchups} matchups | ${overallData.matchup_win_pct}% win rate`;

        // Create month chart
        monthChart = createPieChart(
            'month-chart',
            monthData.matchup_wins,
            monthData.matchup_losses,
            monthData.matchup_ties
        );
        document.getElementById('month-chart-stats').innerHTML =
            `${monthData.month_matchups} matchups | ${monthData.matchup_win_pct}% win rate`;

    } catch (error) {
        console.error('Error loading charts:', error);
    }
}

// Refresh all data
function refreshData() {
    loadTodayStats();
    loadStats();
    loadGames();
    loadMatchups();
    loadCharts();
    updateUndoButton();
}

// Update undo button visibility
function updateUndoButton() {
    const undoBtn = document.getElementById('undo-btn');
    if (lastAction) {
        undoBtn.classList.remove('hidden');
        let actionText = '';
        if (lastAction.type === 'create') actionText = 'Undo Add';
        else if (lastAction.type === 'update') actionText = 'Undo Edit';
        else if (lastAction.type === 'delete') actionText = 'Undo Delete';
        undoBtn.textContent = actionText;
    } else {
        undoBtn.classList.add('hidden');
    }
}

// Undo last action
async function undoLastAction() {
    if (!lastAction) return;

    try {
        if (lastAction.type === 'create') {
            // Undo create = delete the game
            await fetch(`/api/games/${lastAction.gameId}`, { method: 'DELETE' });
        } else if (lastAction.type === 'update') {
            // Undo update = restore previous data
            await fetch(`/api/games/${lastAction.gameId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(lastAction.previousData)
            });
        } else if (lastAction.type === 'delete') {
            // Undo delete = recreate the game
            await fetch('/api/games', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(lastAction.gameData)
            });
        }

        lastAction = null;
        refreshData();
    } catch (error) {
        console.error('Error undoing action:', error);
        alert('Error undoing action');
    }
}

// Modal functions
function openAddModal() {
    document.getElementById('modal-title').textContent = 'Add Game';
    document.getElementById('game-id').value = '';
    document.getElementById('game-form').reset();

    // Set default datetime to now (local time, no UTC conversion)
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    document.getElementById('game-datetime').value = `${year}-${month}-${day}T${hours}:${minutes}`;

    document.getElementById('game-modal').classList.add('active');
}

async function openEditModal(gameId) {
    try {
        const response = await fetch(`/api/games/${gameId}`);
        const game = await response.json();

        console.log('Game datetime from API:', game.datetime);
        console.log('Formatted for input:', formatDateForInput(game.datetime));

        document.getElementById('modal-title').textContent = 'Edit Game';
        document.getElementById('game-id').value = gameId;

        // Populate form fields
        document.getElementById('game-datetime').value = formatDateForInput(game.datetime);
        document.getElementById('game-win').value = game.win;
        document.getElementById('game-opponent').value = game.opponent || '';

        const fields = ['p1_character', 'p2_character', 'p3_character', 'p4_character',
                        'p1_kos', 'p2_kos', 'p3_kos', 'p4_kos',
                        'p1_falls', 'p2_falls', 'p3_falls', 'p4_falls',
                        'p1_damage', 'p2_damage', 'p3_damage', 'p4_damage'];

        fields.forEach(field => {
            const value = game[field];
            // Display null values as empty string (user can leave blank or fill in)
            document.getElementById(`game-${field}`).value = (value === null || value === undefined) ? '' : value;
        });

        document.getElementById('game-modal').classList.add('active');
    } catch (error) {
        console.error('Error loading game:', error);
        alert('Error loading game data');
    }
}

function closeModal() {
    document.getElementById('game-modal').classList.remove('active');
}

async function saveGame(event) {
    event.preventDefault();

    const gameId = document.getElementById('game-id').value;
    const isEdit = !!gameId;

    // Helper to parse stat values - empty string becomes null (displayed as N/A)
    // If user wants 0, they must explicitly type 0
    function parseStat(fieldId) {
        const value = document.getElementById(fieldId).value.trim();
        if (value === '') {
            return null;  // Empty fields become null (displayed as N/A)
        }
        const parsed = parseInt(value);
        return isNaN(parsed) ? null : parsed;
    }

    const gameData = {
        datetime: formatDateFromInput(document.getElementById('game-datetime').value),
        win: document.getElementById('game-win').value,
        opponent: document.getElementById('game-opponent').value,
        p1_character: document.getElementById('game-p1_character').value,
        p2_character: document.getElementById('game-p2_character').value,
        p3_character: document.getElementById('game-p3_character').value,
        p4_character: document.getElementById('game-p4_character').value,
        p1_kos: parseStat('game-p1_kos'),
        p2_kos: parseStat('game-p2_kos'),
        p3_kos: parseStat('game-p3_kos'),
        p4_kos: parseStat('game-p4_kos'),
        p1_falls: parseStat('game-p1_falls'),
        p2_falls: parseStat('game-p2_falls'),
        p3_falls: parseStat('game-p3_falls'),
        p4_falls: parseStat('game-p4_falls'),
        p1_damage: parseStat('game-p1_damage'),
        p2_damage: parseStat('game-p2_damage'),
        p3_damage: parseStat('game-p3_damage'),
        p4_damage: parseStat('game-p4_damage'),
    };

    try {
        // For edits, fetch the current data first for undo
        let previousData = null;
        if (isEdit) {
            const prevResponse = await fetch(`/api/games/${gameId}`);
            previousData = await prevResponse.json();
        }

        const url = isEdit ? `/api/games/${gameId}` : '/api/games';
        const method = isEdit ? 'PUT' : 'POST';

        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gameData)
        });

        if (response.ok) {
            const result = await response.json();

            // Store action for undo
            if (isEdit) {
                lastAction = { type: 'update', gameId: parseInt(gameId), previousData };
            } else {
                lastAction = { type: 'create', gameId: result.id };
            }

            closeModal();
            refreshData();
        } else {
            const error = await response.json();
            alert('Error saving game: ' + (error.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error saving game:', error);
        alert('Error saving game');
    }
}

// Delete modal functions
function openDeleteModal(gameId) {
    document.getElementById('delete-game-id').value = gameId;
    document.getElementById('delete-modal').classList.add('active');
}

function closeDeleteModal() {
    document.getElementById('delete-modal').classList.remove('active');
}

async function confirmDelete() {
    const gameId = document.getElementById('delete-game-id').value;

    try {
        // Fetch game data before deletion for undo
        const gameResponse = await fetch(`/api/games/${gameId}`);
        const gameData = await gameResponse.json();

        const response = await fetch(`/api/games/${gameId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            // Store action for undo (remove id from gameData so it creates a new record)
            delete gameData.id;
            lastAction = { type: 'delete', gameData };

            closeDeleteModal();
            refreshData();
        } else {
            const error = await response.json();
            alert('Error deleting game: ' + (error.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error deleting game:', error);
        alert('Error deleting game');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    refreshData();
});
