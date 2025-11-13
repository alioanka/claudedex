// dashboard/static/js/logs.js

document.addEventListener('DOMContentLoaded', function() {
    const logTableBody = document.getElementById('logTableBody');
    const logSearch = document.getElementById('logSearch');
    const logLevelFilter = document.getElementById('logLevelFilter');

    let logs = [];

    async function fetchLogs() {
        try {
            const response = await apiGet('/api/logs');
            if (response.success) {
                logs = response.data;
                renderLogs();
            }
        } catch (error) {
            console.error('Failed to fetch logs:', error);
            showToast('error', 'Failed to load logs');
        }
    }

    function renderLogs() {
        const searchTerm = logSearch.value.toLowerCase();
        const levelFilter = logLevelFilter.value;

        const filteredLogs = logs.filter(log => {
            const messageMatch = log.message.toLowerCase().includes(searchTerm);
            const levelMatch = levelFilter === 'all' || log.level === levelFilter;
            return messageMatch && levelMatch;
        });

        logTableBody.innerHTML = filteredLogs.map(log => `
            <tr>
                <td>${formatDate(log.timestamp)} ${formatTime(log.timestamp)}</td>
                <td><span class="log-level ${log.level.toLowerCase()}">${log.level}</span></td>
                <td>${log.message}</td>
            </tr>
        `).join('');
    }

    logSearch.addEventListener('input', renderLogs);
    logLevelFilter.addEventListener('change', renderLogs);

    fetchLogs();
    setInterval(fetchLogs, 10000); // Refresh logs every 10 seconds
});
