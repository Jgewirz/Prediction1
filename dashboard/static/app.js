/**
 * Kalshi Trading Bot Dashboard - Client-side JavaScript
 * Version 2.0 - WebSocket Real-time Updates
 */

// =====================
// State Management
// =====================
const state = {
    page: 1,
    perPage: 50,
    totalPages: 1,
    currentPeriod: '7d',
    sortBy: 'timestamp',
    sortOrder: 'desc',
    wsConnected: false,
    reconnectAttempts: 0,
    maxReconnectAttempts: 10,
    decisions: [],
    kpis: null
};

// WebSocket instance
let ws = null;
let reconnectTimeout = null;

// =====================
// Initialization
// =====================
document.addEventListener('DOMContentLoaded', () => {
    initializeWebSocket();
    loadStatus();
    loadKPIs();
    loadDecisions();
    setupEventListeners();
});

// =====================
// WebSocket Management
// =====================
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    console.log('Connecting to WebSocket:', wsUrl);
    updateConnectionStatus('connecting');

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = handleWsOpen;
        ws.onmessage = handleWsMessage;
        ws.onclose = handleWsClose;
        ws.onerror = handleWsError;
    } catch (err) {
        console.error('WebSocket initialization failed:', err);
        updateConnectionStatus('disconnected');
        scheduleReconnect();
    }
}

function handleWsOpen() {
    console.log('WebSocket connected');
    state.wsConnected = true;
    state.reconnectAttempts = 0;
    updateConnectionStatus('connected');

    // Subscribe to all topics including workflow
    ws.send(JSON.stringify({
        action: 'subscribe',
        topics: ['decisions', 'kpis', 'alerts', 'status', 'workflow', 'all']
    }));
}

function handleWsMessage(event) {
    try {
        const message = JSON.parse(event.data);
        console.log('WS Message:', message.type);

        switch (message.type) {
            case 'connected':
                console.log('WebSocket welcomed:', message.data.client_id);
                break;

            case 'subscribed':
                console.log('Subscribed to topics:', message.data.topics);
                break;

            case 'decision':
                handleNewDecision(message.data);
                break;

            case 'kpi_update':
                handleKpiUpdate(message.data);
                break;

            case 'alert':
                handleAlert(message.data);
                break;

            case 'status':
                handleStatusUpdate(message.data);
                break;

            case 'workflow_step':
                handleWorkflowStep(message.data);
                break;

            case 'heartbeat':
                // Connection is alive, update last heartbeat time
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                break;

            case 'error':
                console.error('WS Error:', message.data);
                break;
        }
    } catch (err) {
        console.error('Failed to parse WS message:', err);
    }
}

function handleWsClose(event) {
    console.log('WebSocket closed:', event.code, event.reason);
    state.wsConnected = false;
    updateConnectionStatus('disconnected');
    scheduleReconnect();
}

function handleWsError(error) {
    console.error('WebSocket error:', error);
    state.wsConnected = false;
    updateConnectionStatus('error');
}

function scheduleReconnect() {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
    }

    if (state.reconnectAttempts >= state.maxReconnectAttempts) {
        console.log('Max reconnection attempts reached, falling back to polling');
        startPollingFallback();
        return;
    }

    // Exponential backoff: 1s, 2s, 4s, 8s, ... max 30s
    const delay = Math.min(1000 * Math.pow(2, state.reconnectAttempts), 30000);
    state.reconnectAttempts++;

    console.log(`Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts})`);
    updateConnectionStatus('reconnecting');

    reconnectTimeout = setTimeout(() => {
        initializeWebSocket();
    }, delay);
}

function startPollingFallback() {
    console.log('Starting polling fallback (30s interval)');
    setInterval(() => {
        loadStatus();
        loadKPIs();
        loadDecisions();
    }, 30000);
}

function updateConnectionStatus(status) {
    const indicator = document.getElementById('connection-status');
    if (!indicator) return;

    indicator.className = 'connection-indicator ' + status;

    const labels = {
        'connected': 'Live',
        'connecting': 'Connecting...',
        'disconnected': 'Offline',
        'reconnecting': 'Reconnecting...',
        'error': 'Error'
    };

    indicator.textContent = labels[status] || status;
}

// =====================
// Real-time Handlers
// =====================
function handleNewDecision(decision) {
    console.log('New decision received:', decision.decision_id);

    // Add to the beginning of the list if on page 1
    if (state.page === 1 && state.sortBy === 'timestamp' && state.sortOrder === 'desc') {
        // Check if decision passes current filters
        if (passesFilters(decision)) {
            // Prepend to table
            const tbody = document.getElementById('decisions-body');
            const newRow = createDecisionRow(decision);
            tbody.insertAdjacentHTML('afterbegin', newRow);

            // Highlight the new row briefly
            const firstRow = tbody.querySelector('tr');
            if (firstRow) {
                firstRow.classList.add('new-row');
                setTimeout(() => firstRow.classList.remove('new-row'), 2000);
            }

            // Remove last row if over limit
            const rows = tbody.querySelectorAll('tr:not(.loading-row):not(.empty-row)');
            if (rows.length > state.perPage) {
                rows[rows.length - 1].remove();
            }

            // Update count
            const countEl = document.getElementById('decision-count');
            if (countEl) {
                const currentCount = parseInt(countEl.textContent) || 0;
                countEl.textContent = `${currentCount + 1} decisions`;
            }
        }
    }

    // Show notification toast
    showToast(`New decision: ${decision.action.toUpperCase()} on ${truncate(decision.market_title, 30)}`, 'info');
}

function handleKpiUpdate(kpis) {
    console.log('KPI update received');
    state.kpis = kpis;

    // Update KPI cards
    updateKPI('kpi-pnl', formatCurrency(kpis.realized_pnl), kpis.realized_pnl);
    updateKPI('kpi-winrate', formatPercent(kpis.win_rate));
    updateKPI('kpi-edge', formatPercent(kpis.avg_edge), kpis.avg_edge);
    updateKPI('kpi-rscore', kpis.avg_r_score?.toFixed(2) || '0.00');
    updateKPI('kpi-exposure', formatCurrency(kpis.unrealized_exposure));
    updateKPI('kpi-trades', (kpis.actionable_bets || 0).toString());
    updateKPI('kpi-skipped', (kpis.skip_count || 0).toString());
}

function handleAlert(alert) {
    console.log('Alert received:', alert);

    const severity = alert.severity || 'info';
    const message = alert.message || 'New alert';

    showToast(message, severity);

    // If critical, show persistent notification
    if (severity === 'critical' || severity === 'error') {
        showAlertBanner(alert);
    }
}

function handleStatusUpdate(status) {
    console.log('Status update received:', status);

    if (status.bot_running !== undefined) {
        const botStatus = document.getElementById('bot-status');
        if (botStatus) {
            botStatus.textContent = status.bot_running ? 'Running' : 'Stopped';
            botStatus.className = 'status-indicator ' + (status.bot_running ? 'running' : 'stopped');
        }
    }

    if (status.mode) {
        updateModeDisplay(status.mode);
    }
}

function handleWorkflowStep(step) {
    console.log('Workflow step received:', step);

    const stepNumber = step.step_number;
    const stepName = step.step_name;
    const status = step.status;
    const details = step.details || {};
    const timestamp = step.timestamp;

    // Update workflow status indicator
    const workflowStatus = document.getElementById('workflow-status');
    if (workflowStatus) {
        if (status === 'running') {
            workflowStatus.textContent = 'Running';
            workflowStatus.className = 'workflow-status running';
        } else if (stepNumber === 10 && status === 'completed') {
            workflowStatus.textContent = 'Completed';
            workflowStatus.className = 'workflow-status completed';
        }
    }

    // Find the step element
    const stepEl = document.querySelector(`.workflow-step[data-step="${stepNumber}"]`);
    if (!stepEl) return;

    // Update step class based on status
    stepEl.classList.remove('pending', 'running', 'completed', 'failed', 'skipped');
    stepEl.classList.add(status);

    // Update step icon
    const iconEl = stepEl.querySelector('.step-icon');
    if (iconEl) {
        const icons = {
            'pending': '○',
            'running': '▶',
            'completed': '✓',
            'failed': '✗',
            'skipped': '–'
        };
        iconEl.textContent = icons[status] || '○';
    }

    // Update step name
    const nameEl = stepEl.querySelector('.step-name');
    if (nameEl) {
        nameEl.textContent = stepName;
    }

    // Update step details
    const detailsEl = stepEl.querySelector('.step-details');
    if (detailsEl && Object.keys(details).length > 0) {
        const detailParts = [];
        for (const [key, value] of Object.entries(details)) {
            const label = key.replace(/_/g, ' ');
            detailParts.push(`${label}: ${value}`);
        }
        detailsEl.textContent = detailParts.join(' | ');
    }

    // Update timestamp
    const tsEl = document.getElementById('workflow-timestamp');
    if (tsEl && timestamp) {
        const date = new Date(timestamp);
        tsEl.textContent = `Last update: ${date.toLocaleTimeString()}`;
    }

    // If step 1 is running, reset all other steps to pending
    if (stepNumber === 1 && status === 'running') {
        resetWorkflowSteps();
        // Re-apply running status to step 1
        stepEl.classList.remove('pending');
        stepEl.classList.add('running');
        if (iconEl) iconEl.textContent = '▶';
    }
}

function resetWorkflowSteps() {
    document.querySelectorAll('.workflow-step').forEach(el => {
        el.classList.remove('running', 'completed', 'failed', 'skipped');
        el.classList.add('pending');
        const icon = el.querySelector('.step-icon');
        if (icon) icon.textContent = '○';
        const details = el.querySelector('.step-details');
        if (details) details.textContent = '';
    });
}

function passesFilters(decision) {
    const action = document.getElementById('filter-action').value;
    if (action && action !== 'all' && decision.action !== action) return false;

    const confidence = parseFloat(document.getElementById('filter-confidence').value);
    if (!isNaN(confidence) && decision.confidence < confidence) return false;

    const status = document.getElementById('filter-status').value;
    if (status && status !== 'all' && decision.status !== status) return false;

    const signal = document.getElementById('filter-signal').value;
    if (signal && signal !== 'all' && decision.signal_direction !== signal) return false;

    const search = document.getElementById('filter-search').value.toLowerCase();
    if (search) {
        const title = (decision.event_title || '' + decision.market_title || '').toLowerCase();
        if (!title.includes(search)) return false;
    }

    return true;
}

// =====================
// Event Listeners
// =====================
function setupEventListeners() {
    // Period buttons
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            state.currentPeriod = e.target.dataset.period;
            loadKPIs();
        });
    });

    // Sortable headers
    document.querySelectorAll('.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const newSort = th.dataset.sort;
            if (state.sortBy === newSort) {
                state.sortOrder = state.sortOrder === 'asc' ? 'desc' : 'asc';
            } else {
                state.sortBy = newSort;
                state.sortOrder = 'desc';
            }
            updateSortIndicators();
            loadDecisions();
        });
    });

    // Enter key on search
    document.getElementById('filter-search').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') applyFilters();
    });

    // Modal close on backdrop click
    document.getElementById('reasoning-modal').addEventListener('click', (e) => {
        if (e.target.id === 'reasoning-modal') closeModal();
    });

    // Escape key closes modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

// =====================
// API Calls
// =====================
async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const banner = document.getElementById('status-banner');
        const modeBadge = document.getElementById('mode-badge');
        const botStatus = document.getElementById('bot-status');
        const trendradarStatus = document.getElementById('trendradar-status');
        const lastUpdate = document.getElementById('last-update');

        // Update mode
        if (data.last_run_mode === 'live') {
            banner.classList.remove('dry-run');
            banner.classList.add('live');
            modeBadge.textContent = 'LIVE MODE';
            document.querySelector('.mode-text').textContent = 'Real bets are being placed';
        } else {
            banner.classList.remove('live');
            banner.classList.add('dry-run');
            modeBadge.textContent = 'DRY RUN MODE';
            document.querySelector('.mode-text').textContent = 'No actual bets will be placed';
        }

        // Update status
        botStatus.textContent = data.bot_running ? 'Running' : 'Stopped';
        botStatus.className = 'status-indicator ' + (data.bot_running ? 'running' : 'stopped');

        // Update TrendRadar
        trendradarStatus.textContent = data.trendradar_enabled ? 'Enabled' : 'Disabled';

        // Update last run time
        if (data.last_run_at) {
            const date = new Date(data.last_run_at);
            lastUpdate.textContent = date.toLocaleString();
        } else {
            lastUpdate.textContent = 'Never';
        }
    } catch (err) {
        console.error('Failed to load status:', err);
    }
}

function updateModeDisplay(mode) {
    const banner = document.getElementById('status-banner');
    const modeBadge = document.getElementById('mode-badge');

    if (mode === 'live') {
        banner.classList.remove('dry-run');
        banner.classList.add('live');
        modeBadge.textContent = 'LIVE MODE';
        document.querySelector('.mode-text').textContent = 'Real bets are being placed';
    } else {
        banner.classList.remove('live');
        banner.classList.add('dry-run');
        modeBadge.textContent = 'DRY RUN MODE';
        document.querySelector('.mode-text').textContent = 'No actual bets will be placed';
    }
}

async function loadKPIs() {
    try {
        const response = await fetch(`/api/kpis?period=${state.currentPeriod}`);
        const data = await response.json();

        updateKPI('kpi-pnl', formatCurrency(data.realized_pnl), data.realized_pnl);
        updateKPI('kpi-winrate', formatPercent(data.win_rate));
        updateKPI('kpi-edge', formatPercent(data.avg_edge), data.avg_edge);
        updateKPI('kpi-rscore', data.avg_r_score.toFixed(2));
        updateKPI('kpi-exposure', formatCurrency(data.unrealized_exposure));
        updateKPI('kpi-trades', data.actionable_bets.toString());
        updateKPI('kpi-skipped', data.skip_count.toString());
    } catch (err) {
        console.error('Failed to load KPIs:', err);
    }
}

function updateKPI(id, value, numericValue = null) {
    const el = document.getElementById(id);
    if (!el) return;

    el.textContent = value;

    // Add color class based on value
    el.classList.remove('positive', 'negative');
    if (numericValue !== null) {
        if (numericValue > 0) el.classList.add('positive');
        else if (numericValue < 0) el.classList.add('negative');
    }
}

async function loadDecisions() {
    const tbody = document.getElementById('decisions-body');
    tbody.innerHTML = '<tr class="loading-row"><td colspan="11">Loading decisions...</td></tr>';

    try {
        const params = new URLSearchParams({
            page: state.page,
            per_page: state.perPage,
            sort_by: state.sortBy,
            sort_order: state.sortOrder
        });

        // Add filters
        const action = document.getElementById('filter-action').value;
        if (action && action !== 'all') params.append('action', action);

        const confidence = document.getElementById('filter-confidence').value;
        if (confidence) params.append('min_confidence', confidence);

        const status = document.getElementById('filter-status').value;
        if (status && status !== 'all') params.append('status', status);

        const signal = document.getElementById('filter-signal').value;
        if (signal && signal !== 'all') params.append('signal_direction', signal);

        const search = document.getElementById('filter-search').value;
        if (search) params.append('search', search);

        const response = await fetch(`/api/decisions?${params}`);
        const data = await response.json();

        state.totalPages = data.pages;
        state.decisions = data.decisions;
        updatePagination(data.total);

        if (data.decisions.length === 0) {
            tbody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="11">
                        <div class="empty-state">
                            <h3>No decisions found</h3>
                            <p>Try adjusting your filters or run the trading bot to generate decisions.</p>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = data.decisions.map(d => createDecisionRow(d)).join('');
    } catch (err) {
        console.error('Failed to load decisions:', err);
        tbody.innerHTML = `
            <tr>
                <td colspan="11">
                    <div class="empty-state">
                        <h3>Failed to load decisions</h3>
                        <p>Error: ${err.message}</p>
                    </div>
                </td>
            </tr>
        `;
    }
}

function createDecisionRow(d) {
    const actionClass = d.action.replace('_', '-');
    const actionLabel = d.action.toUpperCase().replace('_', ' ');
    const hedgeBadge = d.is_hedge ? '<span class="hedge-badge">HEDGE</span>' : '';

    const researchPct = d.research_probability !== null
        ? (d.research_probability * 100).toFixed(0) + '%'
        : '--';

    const marketPct = d.calc_market_prob !== null
        ? (d.calc_market_prob * 100).toFixed(0) + '%'
        : '--';

    const rScore = d.r_score !== null ? d.r_score.toFixed(2) : '--';

    const timestamp = d.timestamp ? new Date(d.timestamp).toLocaleString() : '--';

    // Signal badge
    const signalBadge = d.signal_direction
        ? `<span class="signal-badge ${d.signal_direction}">${d.signal_direction.toUpperCase()}</span>`
        : '--';

    return `
        <tr>
            <td>${timestamp}</td>
            <td title="${escapeHtml(d.event_title)}">${truncate(d.event_title, 25)}</td>
            <td title="${escapeHtml(d.market_title)}">${truncate(d.market_title, 30)}</td>
            <td><span class="action-badge ${actionClass}">${actionLabel}</span>${hedgeBadge}</td>
            <td class="numeric">${formatCurrency(d.bet_amount)}</td>
            <td class="numeric">${researchPct}</td>
            <td class="numeric">${marketPct}</td>
            <td class="numeric">${d.confidence.toFixed(2)}</td>
            <td class="numeric ${getRScoreClass(d.r_score)}">${rScore}</td>
            <td>${signalBadge}</td>
            <td class="reasoning-cell" onclick="showDecisionDetail('${d.decision_id}')">${truncate(d.reasoning, 40)}</td>
        </tr>
    `;
}

function getRScoreClass(rScore) {
    if (rScore === null) return 'value-neutral';
    if (rScore >= 1.5) return 'value-positive';
    if (rScore <= -1.5) return 'value-negative';
    return 'value-neutral';
}

// =====================
// Pagination
// =====================
function updatePagination(total) {
    document.getElementById('decision-count').textContent = `${total} decisions`;
    document.getElementById('page-info').textContent = `Page ${state.page} of ${state.totalPages}`;
    document.getElementById('prev-btn').disabled = state.page <= 1;
    document.getElementById('next-btn').disabled = state.page >= state.totalPages;
}

function prevPage() {
    if (state.page > 1) {
        state.page--;
        loadDecisions();
    }
}

function nextPage() {
    if (state.page < state.totalPages) {
        state.page++;
        loadDecisions();
    }
}

// =====================
// Filters
// =====================
function applyFilters() {
    state.page = 1;
    loadDecisions();
}

function clearFilters() {
    document.getElementById('filter-action').value = 'all';
    document.getElementById('filter-confidence').value = '';
    document.getElementById('filter-status').value = 'all';
    document.getElementById('filter-signal').value = 'all';
    document.getElementById('filter-search').value = '';
    state.page = 1;
    loadDecisions();
}

// =====================
// Sorting
// =====================
function updateSortIndicators() {
    document.querySelectorAll('.sortable').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
        if (th.dataset.sort === state.sortBy) {
            th.classList.add(state.sortOrder === 'asc' ? 'sorted-asc' : 'sorted-desc');
        }
    });
}

// =====================
// Modal
// =====================
async function showDecisionDetail(decisionId) {
    try {
        const response = await fetch(`/api/decisions/${decisionId}`);
        const data = await response.json();

        const body = document.getElementById('modal-body');
        body.innerHTML = `
            <div class="detail-row">
                <span class="detail-label">Event:</span>
                <span class="detail-value">${escapeHtml(data.event_title || '--')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Market:</span>
                <span class="detail-value">${escapeHtml(data.market_title || '--')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Action:</span>
                <span class="detail-value">${(data.action || '--').toUpperCase().replace('_', ' ')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Amount:</span>
                <span class="detail-value">${formatCurrency(data.bet_amount)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Confidence:</span>
                <span class="detail-value">${(data.confidence || 0).toFixed(2)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">R-Score:</span>
                <span class="detail-value">${data.r_score ? data.r_score.toFixed(3) : '--'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Research Prob:</span>
                <span class="detail-value">${data.research_probability ? (data.research_probability * 100).toFixed(1) + '%' : '--'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Market Prob:</span>
                <span class="detail-value">${data.calc_market_prob ? (data.calc_market_prob * 100).toFixed(1) + '%' : '--'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Status:</span>
                <span class="detail-value">${data.status || 'pending'}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Signal:</span>
                <span class="detail-value">${data.signal_direction || 'none'} (strength: ${data.signal_strength?.toFixed(2) || '--'})</span>
            </div>
            <div class="reasoning-full">
                <strong>Reasoning:</strong><br><br>
                ${escapeHtml(data.reasoning || 'No reasoning provided.')}
            </div>
        `;

        document.getElementById('reasoning-modal').classList.add('show');
    } catch (err) {
        console.error('Failed to load decision detail:', err);
        alert('Failed to load decision details');
    }
}

function closeModal() {
    document.getElementById('reasoning-modal').classList.remove('show');
}

// =====================
// Toast Notifications
// =====================
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2000;
            display: flex;
            flex-direction: column;
            gap: 8px;
        `;
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        background: var(--bg-secondary, #161b22);
        border: 1px solid var(--border-color, #30363d);
        border-left: 4px solid ${getToastColor(type)};
        color: var(--text-primary, #e6edf3);
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        max-width: 350px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;

    container.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function getToastColor(type) {
    const colors = {
        'info': '#58a6ff',
        'success': '#3fb950',
        'warning': '#d29922',
        'error': '#f85149',
        'critical': '#f85149'
    };
    return colors[type] || colors.info;
}

function showAlertBanner(alert) {
    let banner = document.getElementById('alert-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'alert-banner';
        banner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(248, 81, 73, 0.9);
            color: white;
            padding: 12px 20px;
            text-align: center;
            font-weight: 500;
            z-index: 3000;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 16px;
        `;
        document.body.prepend(banner);
    }

    banner.innerHTML = `
        <span>${alert.message}</span>
        <button onclick="this.parentElement.remove()" style="
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
        ">Dismiss</button>
    `;
}

// =====================
// Utilities
// =====================
function formatCurrency(value) {
    if (value === null || value === undefined) return '$0.00';
    const prefix = value < 0 ? '-' : '';
    return prefix + '$' + Math.abs(value).toFixed(2);
}

function formatPercent(value) {
    if (value === null || value === undefined) return '0%';
    const prefix = value > 0 ? '+' : '';
    return prefix + value.toFixed(1) + '%';
}

function truncate(str, length) {
    if (!str) return '--';
    if (str.length <= length) return escapeHtml(str);
    return escapeHtml(str.substring(0, length)) + '...';
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    .new-row {
        animation: highlight 2s ease;
    }
    @keyframes highlight {
        0% { background: rgba(88, 166, 255, 0.3); }
        100% { background: transparent; }
    }
    .connection-indicator {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .connection-indicator.connected {
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
    }
    .connection-indicator.connecting,
    .connection-indicator.reconnecting {
        background: rgba(210, 153, 34, 0.15);
        color: #d29922;
    }
    .connection-indicator.disconnected,
    .connection-indicator.error {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
    }
`;
document.head.appendChild(style);
