// dashboard/static/js/logs.js
//
// Renders /logs page. Calls /api/logs with operator-tunable filters:
//   ?module=<name>  - one of dex, futures, solana, sniper, arbitrage,
//                     copy_trading, ai, dashboard
//   ?level=ERROR    - INFO / WARNING / ERROR / CRITICAL (server-side
//                     simple substring filter)
//   ?limit=N        - capped at 2000 server-side
// Backend returns {success, data: [{module, file, timestamp, level, message}],
// count}.
//
// Client-side: free-text search inside the message field.

document.addEventListener('DOMContentLoaded', function () {
    const logTableBody = document.getElementById('logTableBody');
    const logSearch = document.getElementById('logSearch');
    const logLevelFilter = document.getElementById('logLevelFilter');
    const logModuleFilter = document.getElementById('logModuleFilter');

    let logs = [];

    async function fetchLogs() {
        try {
            const params = new URLSearchParams();
            params.set('limit', '500');
            const mod = logModuleFilter ? logModuleFilter.value : 'all';
            const level = logLevelFilter ? logLevelFilter.value : 'all';
            if (mod && mod !== 'all') params.set('module', mod);
            if (level && level !== 'all') params.set('level', level);
            const response = await apiGet(`/api/logs?${params.toString()}`);
            if (response && response.success) {
                logs = response.data || [];
                renderLogs();
            }
        } catch (error) {
            console.error('Failed to fetch logs:', error);
            if (typeof showToast === 'function') showToast('error', 'Failed to load logs');
        }
    }

    function renderLogs() {
        const searchTerm = (logSearch.value || '').toLowerCase();
        const filtered = logs.filter(log =>
            (log.message || '').toLowerCase().includes(searchTerm)
        );

        if (filtered.length === 0) {
            logTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted" style="padding:20px;">No logs match the current filters.</td></tr>';
            return;
        }

        logTableBody.innerHTML = filtered.map(log => {
            const lvl = (log.level || 'INFO').toUpperCase();
            return `
                <tr>
                    <td style="white-space:nowrap;">${log.timestamp || ''}</td>
                    <td><span class="badge badge-secondary">${log.module || ''}</span></td>
                    <td><span class="log-level ${lvl.toLowerCase()}">${lvl}</span></td>
                    <td style="font-family:monospace;font-size:0.85rem;">${log.message || ''}</td>
                </tr>
            `;
        }).join('');
    }

    logSearch.addEventListener('input', renderLogs);
    logLevelFilter.addEventListener('change', fetchLogs);  // re-fetch since level is server-side
    if (logModuleFilter) logModuleFilter.addEventListener('change', fetchLogs);

    fetchLogs();
    setInterval(fetchLogs, 30000);  // 30s — was 10s, lighter on the DB
});
