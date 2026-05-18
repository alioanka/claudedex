/* dry_run_widget.js — canonical DRY/LIVE banner injected at the top of
 * every per-module settings page. Talks to /api/modules/{module}/dry-run
 * directly so the operator can flip without filling out the full form.
 *
 * Usage: include this script and call
 *   DryRunWidget.init('arbitrage');     // module key matches _DRY_RUN_CONFIG_TYPE_MAP
 *
 * Module keys: arbitrage | sniper | copy_trading | ai | futures | solana | dex
 */
(function (global) {
    'use strict';

    function readCsrf() {
        if (typeof window.withCsrfHeaders === 'function') {
            return window.withCsrfHeaders('POST');
        }
        // Fallback if main.js hasn't loaded yet.
        const m = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
        const token = m ? decodeURIComponent(m[1]) : '';
        return { 'Content-Type': 'application/json', 'X-CSRF-Token': token };
    }

    function render(module, state) {
        const isDry = !!state.effective_dry_run;
        const chip = isDry
            ? '<span style="background:#fbbf24;color:#111;padding:4px 10px;border-radius:12px;font-weight:700;font-size:0.85em">DRY</span>'
            : '<span style="background:#ef4444;color:#fff;padding:4px 10px;border-radius:12px;font-weight:700;font-size:0.85em">LIVE</span>';
        const dbVal = state.db_value === null
            ? '<em style="opacity:0.7">no override (env / global / default)</em>'
            : `<code>${state.db_value}</code>`;
        return `
            <div class="dry-run-banner" style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px 18px;margin:0 0 18px 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
              <div>
                <strong style="font-size:1.05em">Effective DRY_RUN:</strong> ${chip}
                &nbsp;&nbsp;<span style="opacity:0.85">DB override: ${dbVal}</span>
                <div style="font-size:0.85em;opacity:0.7;margin-top:6px">
                  This is the canonical per-module DRY/LIVE toggle for <code>${module}</code>.
                  Flips persist in the database; the module subprocess must restart for the change to take effect.
                </div>
              </div>
              <div style="display:flex;gap:8px">
                <button class="btn btn-warning" data-flip="true" style="padding:6px 14px;font-weight:600">
                  Switch to DRY
                </button>
                <button class="btn btn-danger" data-flip="false" style="padding:6px 14px;font-weight:600">
                  Switch to LIVE
                </button>
              </div>
            </div>
        `;
    }

    async function fetchState(module) {
        const resp = await fetch(`/api/modules/${module}/dry-run`, {
            credentials: 'same-origin',
        });
        if (!resp.ok) throw new Error(`dry-run GET ${module} returned ${resp.status}`);
        return resp.json();
    }

    async function flipState(module, newValue) {
        const resp = await fetch(`/api/modules/${module}/dry-run`, {
            method: 'POST',
            headers: readCsrf(),
            credentials: 'same-origin',
            body: JSON.stringify({ dry_run: !!newValue }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || data.success === false) {
            const msg = (data && data.error) || `HTTP ${resp.status}`;
            throw new Error(msg);
        }
        return data;
    }

    function bindButtons(container, module) {
        container.querySelectorAll('button[data-flip]').forEach((btn) => {
            btn.addEventListener('click', async () => {
                const newValue = btn.dataset.flip === 'true';
                btn.disabled = true;
                const oldLabel = btn.textContent;
                btn.textContent = 'Saving...';
                try {
                    await flipState(module, newValue);
                    await refresh(container, module);
                    if (typeof window.showToast === 'function') {
                        window.showToast(`Set ${module} dry_run=${newValue}. Restart the subprocess for effect.`, 'success');
                    } else {
                        alert(`Set ${module} dry_run=${newValue}. Restart the subprocess for effect.`);
                    }
                } catch (err) {
                    if (typeof window.showToast === 'function') {
                        window.showToast(`Failed: ${err.message}`, 'error');
                    } else {
                        alert(`Failed to flip ${module} dry_run: ${err.message}`);
                    }
                } finally {
                    btn.disabled = false;
                    btn.textContent = oldLabel;
                }
            });
        });
    }

    async function refresh(container, module) {
        try {
            const state = await fetchState(module);
            container.innerHTML = render(module, state);
            bindButtons(container, module);
        } catch (err) {
            container.innerHTML = `<div class="dry-run-banner" style="background:#7f1d1d;color:#fff;padding:10px 14px;border-radius:8px;margin-bottom:18px">Failed to read DRY_RUN state for <code>${module}</code>: ${err.message}</div>`;
        }
    }

    const DryRunWidget = {
        init: function (module, opts) {
            opts = opts || {};
            const containerId = opts.container_id || 'dry-run-widget';
            let container = document.getElementById(containerId);
            if (!container) {
                container = document.createElement('div');
                container.id = containerId;
                // Insert at top of main content area, before any page header.
                const anchor = document.querySelector('.page-header')
                    || document.querySelector('main')
                    || document.body.firstElementChild;
                if (anchor && anchor.parentNode) {
                    anchor.parentNode.insertBefore(container, anchor);
                } else {
                    document.body.insertBefore(container, document.body.firstChild);
                }
            }
            refresh(container, module);
        },
    };

    global.DryRunWidget = DryRunWidget;
})(window);
