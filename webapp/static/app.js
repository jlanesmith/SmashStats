// SmashStats Frontend JavaScript

// Undo system - stores last action
let lastAction = null;

// Chart instances
let todayChart = null;
let overallChart = null;
let monthChart = null;
let p1CharacterChart = null;
let p2CharacterChart = null;
let p1MonthChart = null;
let p2MonthChart = null;
let p1KoDamageChart = null;
let p2KoDamageChart = null;
let opponentsKoDamageChart = null;
let opponentsCharacterChart = null;
let weekdayChart = null;
let halfhourChart = null;
let dailyChart = null;
let orderChart = null;
let opponentPairsChart = null;
let streaksChart = null;

// Tab switching
function showTab(tabName, updateHash = true) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('tab-active');
        el.classList.add('text-gray-500');
    });

    document.getElementById(`content-${tabName}`).classList.remove('hidden');
    const activeTab = document.getElementById(`tab-${tabName}`);
    activeTab.classList.add('tab-active');
    activeTab.classList.remove('text-gray-500');

    // Update URL hash for tab persistence
    if (updateHash) {
        window.location.hash = tabName;
    }
}

// Get tab from URL hash
function getTabFromHash() {
    const hash = window.location.hash.slice(1); // Remove the #
    const validTabs = ['games', 'matchups', 'graphs', 'coffeetable'];
    return validTabs.includes(hash) ? hash : 'games';
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
    const tbody = document.getElementById('games-table');

    // Show loading state
    tbody.innerHTML = `
        <tr class="loading-row">
            <td colspan="9">
                <div class="loading-spinner"></div>
                <div class="text-gray-500 mt-2">Loading games...</div>
            </td>
        </tr>
    `;

    try {
        const response = await fetch('/api/games');
        const data = await response.json();

        if (data.games.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" class="px-3 py-8 text-center text-gray-500">
                        No games recorded yet. Click "+ Add Game" to get started.
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = data.games.map(game => {
            const team1KOs = formatTeamStat(game.p1_kos, game.p2_kos);
            const team2KOs = formatTeamStat(game.p3_kos, game.p4_kos);
            const teamKOs = `${team1KOs} - ${team2KOs}`;

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
                    <td class="px-3 py-2 whitespace-nowrap">
                        <button onclick="openEditModal(${game.id})" class="text-blue-500 hover:text-blue-700 mr-2">Edit</button>
                        <button onclick="openDeleteModal(${game.id})" class="text-red-500 hover:text-red-700">Delete</button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading games:', error);
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="px-3 py-8 text-center text-red-500">
                    Error loading games. Please refresh the page.
                </td>
            </tr>
        `;
    }
}

// Load and display matchups
async function loadMatchups() {
    const tbody = document.getElementById('matchups-table');

    // Show loading state
    tbody.innerHTML = `
        <tr class="loading-row">
            <td colspan="9">
                <div class="loading-spinner"></div>
                <div class="text-gray-500 mt-2">Loading matchups...</div>
            </td>
        </tr>
    `;

    try {
        const response = await fetch('/api/matchups');
        const data = await response.json();

        if (data.matchups.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" class="px-3 py-8 text-center text-gray-500">
                        No matchups recorded yet.
                    </td>
                </tr>
            `;
            return;
        }

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
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="px-3 py-8 text-center text-red-500">
                    Error loading matchups. Please refresh the page.
                </td>
            </tr>
        `;
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
            ${formatWinLoss(data.matchup_wins)}W - ${formatWinLoss(data.matchup_losses)}L (${data.matchup_win_pct}%)
        `;
    } catch (error) {
        console.error('Error loading today stats:', error);
    }
}

// Format win/loss numbers - only show decimal if it's .5
function formatWinLoss(value) {
    // If it's a whole number, return without decimals
    if (value % 1 === 0) {
        return Math.round(value).toString();
    }
    // Otherwise show one decimal (for .5 values)
    return value.toFixed(1);
}

// Load overall stats
async function loadStats() {
    try {
        const [gameStatsRes, matchupStatsRes] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/stats/overall')
        ]);

        const gameData = await gameStatsRes.json();
        const matchupData = await matchupStatsRes.json();

        document.getElementById('overall-stats').innerHTML = `
            <span class="font-semibold">Overall:</span>
            ${gameData.total_games} g |
            ${matchupData.total_matchups} m |
            ${formatWinLoss(matchupData.matchup_wins)}W - ${formatWinLoss(matchupData.matchup_losses)}L (${matchupData.matchup_win_pct}%)
        `;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Create pie chart helper
// Ties are already counted as 0.5 wins and 0.5 losses in the backend
function createPieChart(canvasId, wins, losses, ties) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const total = wins + losses;

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
        labels: ['Wins', 'Losses'],
        datasets: [{
            data: [wins, losses],
            backgroundColor: ['#22c55e', '#ef4444'],
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
                            return `${context.label}: ${value.toFixed(1)} (${pct}%)`;
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
            `${todayData.today_matchups} matchups | ${todayData.matchup_win_pct}% success rate`;

        // Create overall chart
        overallChart = createPieChart(
            'overall-chart',
            overallData.matchup_wins,
            overallData.matchup_losses,
            overallData.matchup_ties
        );
        document.getElementById('overall-chart-stats').innerHTML =
            `${overallData.total_matchups} matchups | ${overallData.matchup_win_pct}% success rate`;

        // Create month chart
        monthChart = createPieChart(
            'month-chart',
            monthData.matchup_wins,
            monthData.matchup_losses,
            monthData.matchup_ties
        );
        document.getElementById('month-chart-stats').innerHTML =
            `${monthData.month_matchups} matchups | ${monthData.matchup_win_pct}% success rate`;

    } catch (error) {
        console.error('Error loading charts:', error);
    }
}

// Create character stats bar+line chart
function createCharacterChart(canvasId, characterData, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // Handle empty data
    if (!characterData || characterData.length === 0) {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['No Data'],
                datasets: [{
                    data: [0],
                    backgroundColor: ['#e5e7eb']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    const labels = characterData.map(c => c.character);
    const counts = characterData.map(c => c.count);
    const winPcts = characterData.map(c => c.win_pct);

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Matchups',
                    data: counts,
                    backgroundColor: 'rgba(96, 165, 250, 0.8)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1,
                    yAxisID: 'y',
                    order: 2
                },
                {
                    label: 'Success %',
                    data: winPcts,
                    type: 'line',
                    borderColor: 'rgba(220, 38, 38, 1)',
                    backgroundColor: 'rgba(220, 38, 38, 0.1)',
                    borderWidth: 3,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgba(220, 38, 38, 1)',
                    fill: false,
                    yAxisID: 'y1',
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        afterBody: function(context) {
                            const idx = context[0].dataIndex;
                            const char = characterData[idx];
                            let result = `Wins: ${char.wins} | Losses: ${char.losses}`;
                            if (char.ties > 0) {
                                result += ` | Ties: ${char.ties}`;
                            }
                            return result;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Matchups'
                    }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Success %'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Load and render character charts
async function loadCharacterCharts() {
    try {
        const response = await fetch('/api/stats/characters');
        const data = await response.json();

        // Destroy existing charts if they exist
        if (p1CharacterChart) p1CharacterChart.destroy();
        if (p2CharacterChart) p2CharacterChart.destroy();

        // Create P1 character chart
        p1CharacterChart = createCharacterChart('p1-characters-chart', data.p1, 'P1 Characters');

        // Create P2 character chart
        p2CharacterChart = createCharacterChart('p2-characters-chart', data.p2, 'P2 Characters');

    } catch (error) {
        console.error('Error loading character charts:', error);
    }
}

// Load and render monthly character charts
async function loadMonthCharacterCharts() {
    try {
        const response = await fetch('/api/stats/characters/month');
        const data = await response.json();

        // Destroy existing charts if they exist
        if (p1MonthChart) p1MonthChart.destroy();
        if (p2MonthChart) p2MonthChart.destroy();

        // Create P1 month character chart
        p1MonthChart = createCharacterChart('p1-month-chart', data.p1, 'P1 Last 30 Days');

        // Create P2 month character chart
        p2MonthChart = createCharacterChart('p2-month-chart', data.p2, 'P2 Last 30 Days');

    } catch (error) {
        console.error('Error loading monthly character charts:', error);
    }
}

// Create a KO/Damage chart for a player
function createKoDamageChart(canvasId, characterData, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (!characterData || characterData.length === 0) {
        return new Chart(ctx, {
            type: 'bar',
            data: { labels: ['No Data'], datasets: [{ data: [0] }] },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    const labels = characterData.map(d => d.character);
    const kos = characterData.map(d => d.avg_kos);
    const damage = characterData.map(d => d.avg_damage);
    const counts = characterData.map(d => d.count);

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Count',
                    data: counts,
                    type: 'line',
                    borderColor: 'rgba(34, 197, 94, 1)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 3,
                    pointRadius: 4,
                    pointBackgroundColor: 'rgba(34, 197, 94, 1)',
                    fill: false,
                    yAxisID: 'y2',
                    order: 0
                },
                {
                    label: 'Avg KOs',
                    data: kos,
                    backgroundColor: 'rgba(96, 165, 250, 0.8)',
                    borderColor: 'rgba(96, 165, 250, 1)',
                    borderWidth: 1,
                    yAxisID: 'y',
                    order: 1
                },
                {
                    label: 'Avg Damage',
                    data: damage,
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: 'rgba(239, 68, 68, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1',
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'KOs'
                    }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    grid: {
                        drawOnChartArea: false
                    },
                    title: {
                        display: true,
                        text: 'Damage'
                    }
                },
                y2: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    display: false
                }
            }
        }
    });
}

// Load and render KO/Damage charts
async function loadKoDamageCharts() {
    try {
        const response = await fetch('/api/stats/characters/ko-damage');
        const data = await response.json();

        // Destroy existing charts if they exist
        if (p1KoDamageChart) p1KoDamageChart.destroy();
        if (p2KoDamageChart) p2KoDamageChart.destroy();

        // Create P1 KO/Damage chart
        p1KoDamageChart = createKoDamageChart('p1-ko-damage-chart', data.p1, 'P1 KOs & Damage');

        // Create P2 KO/Damage chart
        p2KoDamageChart = createKoDamageChart('p2-ko-damage-chart', data.p2, 'P2 KOs & Damage');

    } catch (error) {
        console.error('Error loading KO/Damage charts:', error);
    }
}

// Load and render opponents character chart
async function loadOpponentsCharacterChart() {
    try {
        const response = await fetch('/api/stats/characters/opponents');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (opponentsCharacterChart) opponentsCharacterChart.destroy();

        // Create opponents character chart
        opponentsCharacterChart = createCharacterChart('opponents-characters-chart', data, 'Opponents Characters');

    } catch (error) {
        console.error('Error loading opponents character chart:', error);
    }
}

// Load and render opponents KO/Damage chart
async function loadOpponentsKoDamageChart() {
    try {
        const response = await fetch('/api/stats/characters/opponents/ko-damage');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (opponentsKoDamageChart) opponentsKoDamageChart.destroy();

        // Create opponents KO/Damage chart
        opponentsKoDamageChart = createKoDamageChart('opponents-ko-damage-chart', data, 'Opponents KOs & Damage');

    } catch (error) {
        console.error('Error loading opponents KO/Damage chart:', error);
    }
}

// Load and render weekday performance chart
async function loadWeekdayChart() {
    try {
        const response = await fetch('/api/stats/weekday');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (weekdayChart) weekdayChart.destroy();

        const ctx = document.getElementById('weekday-chart').getContext('2d');

        const labels = data.map(d => d.day);
        const counts = data.map(d => d.count);
        const winPcts = data.map(d => d.win_pct);

        weekdayChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Count',
                        data: counts,
                        backgroundColor: 'rgba(96, 165, 250, 0.8)',
                        borderColor: 'rgba(96, 165, 250, 1)',
                        borderWidth: 1,
                        yAxisID: 'y',
                        order: 1
                    },
                    {
                        label: 'Success %',
                        data: winPcts,
                        type: 'line',
                        borderColor: 'rgba(185, 80, 70, 1)',
                        backgroundColor: 'rgba(185, 80, 70, 0.1)',
                        borderWidth: 3,
                        pointBackgroundColor: 'rgba(185, 80, 70, 1)',
                        pointRadius: 5,
                        fill: false,
                        tension: 0.1,
                        yAxisID: 'y1',
                        order: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Day'
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        beginAtZero: true,
                        title: {
                            display: false
                        }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 100,
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            },
            plugins: [{
                id: 'percentLabels',
                afterDatasetsDraw: function(chart) {
                    const ctx = chart.ctx;
                    const dataset = chart.data.datasets[1]; // Line dataset
                    const meta = chart.getDatasetMeta(1);

                    ctx.save();
                    ctx.font = 'bold 12px Arial';
                    ctx.fillStyle = 'rgba(185, 80, 70, 1)';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'bottom';

                    meta.data.forEach((point, index) => {
                        const value = dataset.data[index];
                        if (value > 0) {
                            ctx.fillText(value + '%', point.x, point.y - 8);
                        }
                    });

                    ctx.restore();
                }
            }]
        });

    } catch (error) {
        console.error('Error loading weekday chart:', error);
    }
}

// Load and render half-hour performance chart
async function loadHalfhourChart() {
    try {
        const response = await fetch('/api/stats/halfhour');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (halfhourChart) halfhourChart.destroy();

        const ctx = document.getElementById('halfhour-chart').getContext('2d');

        const labels = data.map(d => d.time);
        const counts = data.map(d => d.count);
        const successRates = data.map(d => d.success_rate);

        halfhourChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Count',
                        data: counts,
                        backgroundColor: 'rgba(96, 165, 250, 0.8)',
                        borderColor: 'rgba(96, 165, 250, 1)',
                        borderWidth: 1,
                        yAxisID: 'y',
                        order: 1
                    },
                    {
                        label: 'Success %',
                        data: successRates,
                        type: 'line',
                        borderColor: 'rgba(185, 80, 70, 1)',
                        backgroundColor: 'rgba(185, 80, 70, 0.1)',
                        borderWidth: 3,
                        pointBackgroundColor: 'rgba(185, 80, 70, 1)',
                        pointRadius: 4,
                        fill: false,
                        tension: 0.1,
                        yAxisID: 'y1',
                        order: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        beginAtZero: true,
                        title: {
                            display: false
                        }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 100,
                        grid: {
                            drawOnChartArea: false
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading half-hour chart:', error);
    }
}

// Calculate linear regression trendline
function calculateTrendline(data) {
    const n = data.length;
    if (n < 2) return data.map(() => data[0] || 0);

    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (let i = 0; i < n; i++) {
        sumX += i;
        sumY += data[i];
        sumXY += i * data[i];
        sumXX += i * i;
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return data.map((_, i) => slope * i + intercept);
}

// Load and render daily success rate chart
async function loadDailyChart() {
    try {
        const response = await fetch('/api/stats/daily');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (dailyChart) dailyChart.destroy();

        const ctx = document.getElementById('daily-chart').getContext('2d');

        // Format dates as MM/DD/YYYY
        const labels = data.map(d => {
            const [year, month, day] = d.date.split('-');
            return `${month}/${day}/${year}`;
        });
        const successRates = data.map(d => d.success_rate);
        const trendline = calculateTrendline(successRates);

        dailyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Success %',
                        data: successRates,
                        borderColor: 'rgba(96, 165, 250, 1)',
                        backgroundColor: 'rgba(96, 165, 250, 0.1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(96, 165, 250, 1)',
                        pointRadius: 3,
                        fill: false,
                        tension: 0,
                        order: 0
                    },
                    {
                        label: 'Trendline',
                        data: trendline,
                        borderColor: 'rgba(34, 197, 94, 1)',
                        backgroundColor: 'transparent',
                        borderWidth: 3,
                        pointRadius: 0,
                        fill: false,
                        tension: 0,
                        order: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Success %'
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading daily chart:', error);
    }
}

// Load and render order of games chart
async function loadOrderChart() {
    try {
        const response = await fetch('/api/stats/order');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (orderChart) orderChart.destroy();

        const ctx = document.getElementById('order-chart').getContext('2d');

        const labels = data.map(d => d.pattern);
        const counts = data.map(d => d.count);

        orderChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Count',
                    data: counts,
                    backgroundColor: 'rgba(96, 165, 250, 0.8)',
                    borderColor: 'rgba(96, 165, 250, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Order'
                        },
                        ticks: {
                            maxRotation: 90,
                            minRotation: 90
                        }
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                }
            },
            plugins: [{
                id: 'countLabels',
                afterDatasetsDraw: function(chart) {
                    const ctx = chart.ctx;
                    const dataset = chart.data.datasets[0];
                    const meta = chart.getDatasetMeta(0);

                    ctx.save();
                    ctx.font = 'bold 10px Arial';
                    ctx.fillStyle = 'rgba(96, 165, 250, 1)';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'bottom';

                    meta.data.forEach((bar, index) => {
                        const value = dataset.data[index];
                        ctx.fillText(value, bar.x, bar.y - 2);
                    });

                    ctx.restore();
                }
            }]
        });

    } catch (error) {
        console.error('Error loading order chart:', error);
    }
}

// Load and render opponent pairs chart
async function loadOpponentPairsChart() {
    try {
        const response = await fetch('/api/stats/opponent-pairs');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (opponentPairsChart) opponentPairsChart.destroy();

        const ctx = document.getElementById('opponent-pairs-chart').getContext('2d');

        const labels = data.map(d => d.pair);
        const counts = data.map(d => d.count);
        const successRates = data.map(d => d.success_rate);

        opponentPairsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Count',
                        data: counts,
                        backgroundColor: 'rgba(96, 165, 250, 0.8)',
                        borderColor: 'rgba(96, 165, 250, 1)',
                        borderWidth: 1,
                        xAxisID: 'x'
                    },
                    {
                        label: 'Success %',
                        data: successRates,
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 1,
                        xAxisID: 'x1'
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    },
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        beginAtZero: true,
                        title: {
                            display: false
                        }
                    },
                    x1: {
                        type: 'linear',
                        position: 'top',
                        min: 0,
                        max: 100,
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            },
            plugins: [{
                id: 'barLabels',
                afterDatasetsDraw: function(chart) {
                    const ctx = chart.ctx;

                    // Draw count labels (blue)
                    const countDataset = chart.data.datasets[0];
                    const countMeta = chart.getDatasetMeta(0);
                    ctx.save();
                    ctx.font = 'bold 9px Arial';
                    ctx.fillStyle = 'rgba(96, 165, 250, 1)';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    countMeta.data.forEach((bar, index) => {
                        const value = countDataset.data[index];
                        ctx.fillText(value, bar.x + 3, bar.y);
                    });
                    ctx.restore();

                    // Draw success rate labels (red)
                    const successDataset = chart.data.datasets[1];
                    const successMeta = chart.getDatasetMeta(1);
                    ctx.save();
                    ctx.font = 'bold 9px Arial';
                    ctx.fillStyle = 'rgba(239, 68, 68, 1)';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    successMeta.data.forEach((bar, index) => {
                        const value = successDataset.data[index];
                        ctx.fillText(value, bar.x + 3, bar.y);
                    });
                    ctx.restore();
                }
            }]
        });

    } catch (error) {
        console.error('Error loading opponent pairs chart:', error);
    }
}

// Load and render streaks chart
async function loadStreaksChart() {
    try {
        const response = await fetch('/api/stats/streaks');
        const data = await response.json();

        // Destroy existing chart if it exists
        if (streaksChart) streaksChart.destroy();

        const ctx = document.getElementById('streaks-chart').getContext('2d');

        streaksChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Win Streak', 'Loss Streak'],
                datasets: [{
                    data: [data.max_win_streak, data.max_loss_streak],
                    backgroundColor: ['#22c55e', '#ef4444'],
                    borderColor: ['#16a34a', '#dc2626'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw} matchups`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });

    } catch (error) {
        console.error('Error loading streaks chart:', error);
    }
}

// Refresh all data
function refreshData() {
    loadTodayStats();
    loadStats();
    loadGames();
    loadMatchups();
    loadCharts();
    loadCharacterCharts();
    loadMonthCharacterCharts();
    loadKoDamageCharts();
    loadOpponentsCharacterChart();
    loadOpponentsKoDamageChart();
    loadWeekdayChart();
    loadHalfhourChart();
    loadDailyChart();
    loadOrderChart();
    loadOpponentPairsChart();
    loadStreaksChart();
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
    // Restore tab from URL hash
    const savedTab = getTabFromHash();
    showTab(savedTab, false);

    refreshData();
});

// Handle browser back/forward navigation
window.addEventListener('hashchange', () => {
    const tab = getTabFromHash();
    showTab(tab, false);
});
