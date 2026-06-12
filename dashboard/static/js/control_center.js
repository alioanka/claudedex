/* Control Center v4 — unified module overview, cross-module performance,
 * read-only meta-controller surface.
 *
 * Design rules:
 *  - ONE batched fetch per panel (no per-module N+1 from the browser).
 *  - Fail-soft: fetch errors show a retry state; empty data shows an
 *    empty state; the meta panel hides itself when the table is absent.
 *  - PnL units differ per module — values always render with their unit
 *    and are never summed across modules.
 */
(function () {
    'use strict';

    // module key -> control-endpoint spellings the backend accepts.
    // enable/disable exists only for the 3 env-mapped modules (and is
    // in-process only — MB-33); restart is unsupported for polymarket.
    var CONTROLS = {
        dex:          { pause: 'dex',          restart: 'dex',          dry: 'dex' },
        futures:      { pause: 'futures',      restart: 'futures',      dry: 'futures' },
        solana:       { pause: 'solana',       restart: 'solana',       dry: 'solana' },
        sniper:       { pause: 'sniper',       restart: 'sniper',       dry: 'sniper' },
        arbitrage:    { pause: 'arbitrage',    restart: 'arbitrage',    dry: 'arbitrage' },
        copy_trading: { pause: 'copy_trading', restart: 'copy_trading', dry: 'copy_trading' },
        ai:           { pause: 'ai',           restart: 'ai',           dry: 'ai' },
        polymarket:   { pause: 'polymarket' }
    };

    var STATUS_LABELS = {
        live: 'LIVE', dry_run: 'DRY RUN', paused: 'PAUSED',
        offline: 'OFFLINE', disabled: 'DISABLED',
        killswitch: 'KILLSWITCH', unknown: 'UNKNOWN'
    };

    var MODULE_COLORS = {
        dex: '#3b82f6', futures: '#f59e0b', solana: '#a855f7',
        sniper: '#ef4444', arbitrage: '#10b981', copy_trading: '#06b6d4',
        ai: '#eab308', polymarket: '#ec4899'
    };

    var perfChart = null;
    var perfCache = {};
    var currentDays = 7;

    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    function fmtPnl(v, unit) {
        if (v === null || v === undefined || isNaN(v)) { return '—'; }
        var n = Number(v);
        var s = Math.abs(n) >= 1000 ? n.toFixed(0) : n.toFixed(Math.abs(n) < 1 ? 4 : 2);
        return (n > 0 ? '+' : '') + s + ' ' + unit;
    }

    function pnlClass(v) {
        if (v === null || v === undefined || isNaN(v) || Number(v) === 0) { return ''; }
        return Number(v) > 0 ? 'pos' : 'neg';
    }

    function fmtNum(v, dp) {
        if (v === null || v === undefined || isNaN(v)) { return '—'; }
        return Number(v).toFixed(dp === undefined ? 2 : dp);
    }

    // ---------- 1. Unified overview ----------

    function renderOverview(data) {
        var grid = document.getElementById('cc-grid');
        var state = document.getElementById('cc-overview-state');
        var ks = document.getElementById('cc-killswitch');
        ks.style.display = data.killswitch ? 'block' : 'none';

        var mods = data.modules || [];
        mods.forEach(function (m) { overviewStatuses[m.key] = m; });
        if (!mods.length) {
            state.textContent = 'No module data available.';
            state.style.display = 'block';
            grid.style.display = 'none';
            return;
        }
        var html = mods.map(function (m) {
            var status = m.status || 'unknown';
            var badge = '<span class="cc-badge ' + esc(status) + '">' +
                (STATUS_LABELS[status] || esc(status)) + '</span>';
            var modeBadge = '';
            if (status !== 'disabled' && status !== 'unknown') {
                modeBadge = m.dry_run
                    ? '<span class="cc-badge dry_run" title="No live broadcasts">DRY</span>'
                    : '<span class="cc-badge live" title="Live capital at risk">LIVE</span>';
            }
            var pnlCells;
            if (m.pnl_available === false) {
                // shadow module: no realized PnL — show honest activity stats
                pnlCells =
                    metric('Recorded', m.trades_closed, '') +
                    metric('Simulated', m.sim_trades === undefined ? '—' : m.sim_trades, '') +
                    metric('Avg edge', m.avg_edge_bps !== undefined ? fmtNum(m.avg_edge_bps, 1) + ' bps' : '—', '');
            } else {
                pnlCells =
                    metric('Today', fmtPnl(m.pnl_today, m.unit), pnlClass(m.pnl_today)) +
                    metric('7D', fmtPnl(m.pnl_7d, m.unit), pnlClass(m.pnl_7d)) +
                    metric('All', fmtPnl(m.pnl_all, m.unit), pnlClass(m.pnl_all));
            }
            var statCells =
                metric('Win rate', m.win_rate === null || m.win_rate === undefined ? '—' : fmtNum(m.win_rate, 1) + '%', '') +
                metric('Closed', m.trades_closed, '') +
                metric('Open', m.open_positions === null || m.open_positions === undefined ? '—' : m.open_positions, '');
            return (
                '<div class="cc-card" data-module="' + esc(m.key) + '">' +
                  '<div class="cc-card-head">' +
                    '<span class="name">' + esc(m.name) + '</span>' +
                    '<span class="cc-badges">' + badge + modeBadge + '</span>' +
                  '</div>' +
                  '<div class="cc-metrics">' + pnlCells + '</div>' +
                  '<div class="cc-metrics">' + statCells + '</div>' +
                  '<div class="cc-actions">' + actionButtons(m) + '</div>' +
                '</div>'
            );
        }).join('');
        grid.innerHTML = html;
        grid.style.display = 'grid';
        state.style.display = 'none';
        bindActions(grid);
    }

    function metric(lbl, val, cls) {
        return '<div class="cc-metric"><span class="lbl">' + esc(lbl) +
            '</span><span class="val ' + cls + '">' + val + '</span></div>';
    }

    function actionButtons(m) {
        var c = CONTROLS[m.key] || {};
        var b = [];
        if (c.pause) {
            if (m.paused) {
                b.push(btn('resume', c.pause, 'ok', 'fa-play', 'Resume',
                    'Clear logs/.pause_' + c.pause + ' — module resumes live gating'));
            } else {
                b.push(btn('pause', c.pause, 'warn', 'fa-pause', 'Pause',
                    'Write logs/.pause_* flag — engine halts new live writes within one loop'));
            }
        }
        if (c.restart) {
            b.push(btn('restart', c.restart, '', 'fa-redo', 'Restart',
                'Drop logs/.restart_* flag — orchestrator restarts the subprocess within ~5s'));
        }
        if (c.dry) {
            b.push(btn('dry', c.dry, '', m.dry_run ? 'fa-bolt' : 'fa-shield-alt',
                m.dry_run ? 'Go LIVE' : 'Go DRY',
                'Flips the DB dry_run flag. Takes effect after the module restarts.'));
        }
        return b.join('');
    }

    function btn(action, target, cls, icon, label, title) {
        return '<button class="' + cls + '" data-action="' + action +
            '" data-target="' + esc(target) + '" title="' + esc(title) + '">' +
            '<i class="fas ' + icon + '"></i>' + esc(label) + '</button>';
    }

    function bindActions(grid) {
        grid.querySelectorAll('button[data-action]').forEach(function (el) {
            el.addEventListener('click', function () {
                var action = el.getAttribute('data-action');
                var target = el.getAttribute('data-target');
                var card = el.closest('.cc-card');
                var modKey = card ? card.getAttribute('data-module') : target;
                runAction(action, target, modKey, el);
            });
        });
    }

    function runAction(action, target, modKey, el) {
        var url = null, body = null, confirmMsg = null;
        if (action === 'pause') {
            url = '/api/modules/' + target + '/pause';
        } else if (action === 'resume') {
            url = '/api/modules/' + target + '/start';
        } else if (action === 'restart') {
            url = '/api/modules/' + target + '/restart';
            confirmMsg = 'Restart ' + modKey + '? The orchestrator will respawn the subprocess.';
        } else if (action === 'dry') {
            var goingLive = el.textContent.indexOf('LIVE') !== -1;
            confirmMsg = goingLive
                ? 'Switch ' + modKey + ' to LIVE mode? Real capital will be at risk after the module restarts.'
                : 'Switch ' + modKey + ' to DRY RUN? Takes effect after the module restarts.';
            url = '/api/modules/' + target + '/dry-run';
            body = JSON.stringify({ dry_run: !goingLive });
        }
        if (!url) { return; }
        if (confirmMsg && !window.confirm(confirmMsg)) { return; }
        el.disabled = true;
        fetch(url, {
            method: 'POST',
            headers: window.withCsrfHeaders ? window.withCsrfHeaders('POST') : { 'Content-Type': 'application/json' },
            body: body
        }).then(function (r) { return r.json().catch(function () { return {}; }); })
          .then(function (data) {
              if (data && data.note) { console.info('[control-center]', data.note); }
              return loadOverview();
          })
          .catch(function (e) { console.error('[control-center] action failed', e); })
          .then(function () { el.disabled = false; });
    }

    function loadOverview() {
        return fetch('/api/control-center/overview')
            .then(function (r) {
                if (!r.ok) { throw new Error('HTTP ' + r.status); }
                return r.json();
            })
            .then(renderOverview)
            .catch(function (e) {
                var state = document.getElementById('cc-overview-state');
                state.className = 'cc-state error';
                state.innerHTML = 'Could not load module overview (' + esc(e.message) +
                    ') <button type="button" id="cc-retry-overview">Retry</button>';
                state.style.display = 'block';
                var rb = document.getElementById('cc-retry-overview');
                if (rb) { rb.addEventListener('click', function () {
                    state.className = 'cc-state';
                    state.textContent = 'Loading modules…';
                    loadOverview();
                }); }
            });
    }

    // ---------- 2. Cross-module performance ----------

    function loadPerformance(days) {
        currentDays = days;
        var state = document.getElementById('cc-perf-state');
        var body = document.getElementById('cc-perf-body');
        if (perfCache[days]) {
            renderPerformance(perfCache[days]);
            return;
        }
        state.className = 'cc-state';
        state.textContent = 'Loading performance…';
        state.style.display = 'block';
        body.style.display = 'none';
        fetch('/api/performance/cross-module?days=' + days)
            .then(function (r) {
                if (!r.ok) { throw new Error('HTTP ' + r.status); }
                return r.json();
            })
            .then(function (data) {
                perfCache[days] = data;
                if (data.days === currentDays || (data.days === 0 && currentDays === 0)) {
                    renderPerformance(data);
                }
            })
            .catch(function (e) {
                state.className = 'cc-state error';
                state.textContent = 'Could not load performance (' + e.message + ').';
            });
    }

    function renderPerformance(data) {
        var state = document.getElementById('cc-perf-state');
        var body = document.getElementById('cc-perf-body');
        var tbody = document.querySelector('#cc-perf-table tbody');
        var mods = (data.modules || []);
        var anyTrades = mods.some(function (m) { return (m.trades || 0) > 0; });
        if (!anyTrades) {
            state.className = 'cc-state';
            state.textContent = 'No closed trades in this range yet.';
            state.style.display = 'block';
            body.style.display = 'none';
            return;
        }
        tbody.innerHTML = mods.map(function (m) {
            var mode = overviewMode(m.key);
            if (m.pnl_available === false) {
                return '<tr>' +
                    '<td>' + esc(m.name) + '</td>' +
                    '<td>SHADOW</td>' +
                    '<td class="num">' + (m.trades || 0) + '</td>' +
                    '<td class="num" colspan="5">simulated only — avg expected edge ' +
                        (m.avg_edge_bps !== undefined ? fmtNum(m.avg_edge_bps, 1) + ' bps' : '—') + '</td>' +
                    '<td class="num">—</td>' +
                '</tr>';
            }
            return '<tr>' +
                '<td>' + esc(m.name) + '</td>' +
                '<td>' + mode + '</td>' +
                '<td class="num">' + (m.trades || 0) + '</td>' +
                '<td class="num ' + pnlClass(m.total_pnl) + '">' + fmtPnl(m.total_pnl, m.unit) + '</td>' +
                '<td class="num">' + (m.win_rate === null ? '—' : fmtNum(m.win_rate, 1) + '%') + '</td>' +
                '<td class="num ' + pnlClass(m.expectancy) + '">' + fmtPnl(m.expectancy, m.unit) + '</td>' +
                '<td class="num">' + fmtNum(m.profit_factor, 2) + '</td>' +
                '<td class="num">' + (m.max_drawdown === null ? '—' : fmtNum(m.max_drawdown, 4) + ' ' + esc(m.unit)) + '</td>' +
                '<td class="num">' + fmtNum(m.sharpe_per_trade, 2) + '</td>' +
            '</tr>';
        }).join('');
        state.style.display = 'none';
        body.style.display = 'block';
        renderChart(mods);
    }

    var overviewStatuses = {};
    function overviewMode(key) {
        var st = overviewStatuses[key];
        if (!st) { return '—'; }
        if (st.status === 'disabled') { return 'DISABLED'; }
        return st.dry_run ? 'DRY' : 'LIVE';
    }

    function renderChart(mods) {
        var canvas = document.getElementById('cc-perf-chart');
        if (!canvas || typeof Chart === 'undefined') { return; }
        var datasets = [];
        mods.forEach(function (m) {
            var curve = m.equity_curve || [];
            if (!curve.length) { return; }
            datasets.push({
                label: m.name + ' (' + m.unit + ')',
                data: curve.map(function (p) {
                    return { x: Date.parse(p[0]), y: p[1] };
                }),
                borderColor: MODULE_COLORS[m.key] || '#94a3b8',
                backgroundColor: 'transparent',
                pointRadius: 0,
                borderWidth: 2,
                tension: 0.15
            });
        });
        if (perfChart) { perfChart.destroy(); perfChart = null; }
        if (!datasets.length) { return; }
        perfChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'nearest', intersect: false },
                plugins: {
                    legend: { labels: { color: '#94a3b8', boxWidth: 14 } },
                    tooltip: {
                        callbacks: {
                            title: function (items) {
                                if (!items.length) { return ''; }
                                return new Date(items[0].parsed.x).toLocaleString();
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        ticks: {
                            color: '#64748b',
                            maxTicksLimit: 8,
                            callback: function (v) {
                                return new Date(v).toLocaleDateString();
                            }
                        },
                        grid: { color: 'rgba(148,163,184,0.08)' }
                    },
                    y: {
                        ticks: { color: '#64748b' },
                        grid: { color: 'rgba(148,163,184,0.08)' },
                        title: {
                            display: true, color: '#64748b',
                            text: 'Cumulative PnL (per-module units — shapes comparable, magnitudes not)'
                        }
                    }
                }
            }
        });
    }

    // ---------- 3. Meta controller surface ----------

    function loadMeta() {
        fetch('/api/meta/decisions')
            .then(function (r) {
                if (!r.ok) { throw new Error('HTTP ' + r.status); }
                return r.json();
            })
            .then(function (data) {
                var section = document.getElementById('cc-meta-section');
                if (!data.available || !(data.latest_by_module || []).length) {
                    section.style.display = 'none';
                    return;
                }
                var tbody = document.querySelector('#cc-meta-table tbody');
                tbody.innerHTML = data.latest_by_module.map(function (d) {
                    var dec = String(d.decision || '').toLowerCase();
                    var cls = (dec === 'activate' || dec === 'keep') ? dec
                        : (dec === 'pause' || dec === 'deactivate') ? dec : 'other';
                    var when = '';
                    if (d.created_at) {
                        when = (typeof window.formatTimeAgo === 'function')
                            ? window.formatTimeAgo(d.created_at)
                            : new Date(d.created_at).toLocaleString();
                    }
                    return '<tr>' +
                        '<td>' + esc(d.module) + '</td>' +
                        '<td><span class="cc-meta-decision ' + cls + '">' + esc(d.decision) + '</span></td>' +
                        '<td class="num">' + (d.health_score === null ? '—' : fmtNum(d.health_score, 2)) + '</td>' +
                        '<td class="num">' + (d.confidence === null ? '—' : fmtNum(d.confidence, 2)) + '</td>' +
                        '<td style="white-space:normal; max-width:380px; text-align:left;">' + esc(d.reason) + '</td>' +
                        '<td>' + esc(when) + '</td>' +
                    '</tr>';
                }).join('');
                section.style.display = 'block';
            })
            .catch(function () {
                // fail-soft: panel stays hidden
                var section = document.getElementById('cc-meta-section');
                if (section) { section.style.display = 'none'; }
            });
    }

    // ---------- init ----------

    document.addEventListener('DOMContentLoaded', function () {
        // renderOverview fills overviewStatuses, which the performance
        // table's Mode column reads — so chain perf after overview.
        loadOverview().then(function () { loadPerformance(7); });
        loadMeta();

        document.querySelectorAll('#cc-range button').forEach(function (el) {
            el.addEventListener('click', function () {
                document.querySelectorAll('#cc-range button').forEach(function (b) {
                    b.classList.remove('active');
                });
                el.classList.add('active');
                loadPerformance(parseInt(el.getAttribute('data-days'), 10) || 0);
            });
        });

        // light auto-refresh (overview only — cheap batched endpoint)
        setInterval(function () {
            fetch('/api/control-center/overview')
                .then(function (r) { return r.ok ? r.json() : null; })
                .then(function (data) { if (data) { renderOverview(data); } })
                .catch(function () {});
        }, 30000);
    });
})();
