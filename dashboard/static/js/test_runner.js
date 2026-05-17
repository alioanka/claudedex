/* test_runner.js — frontend for /test-runner.
 *
 * Agent 3 owns the backend contract:
 *   GET  /api/test-runner/tests           → [{id, title, category, kind, cmd_preview, timeout_s}, ...]
 *   POST /api/test-runner/run {test_id}   → {success, exit_code, stdout, stderr, duration_ms}
 *   GET  /api/test-runner/probe/<id>      → {status, body, elapsed_ms}
 *
 * If anything 404s here, that means Agent 3 hasn't shipped the route yet
 * — the page falls back to a clear "endpoint not available" chip so the
 * operator knows the backend half is missing, not the frontend.
 */

(function () {
  'use strict';

  // ---- shared state: every section pushes a markdown chunk here so
  //      "Copy All Results" can stitch a single paste-back blob.
  const RESULTS = {
    sections: {}, // {section_id: [{title, status, body}]}
  };

  // ---- CSRF helper. Use the global if main.js already exposed it,
  //      else fall back to the same cookie-read pattern used in
  //      pro_controls.html / global_settings.html.
  function csrfHeaders(method) {
    if (typeof window.withCsrfHeaders === 'function') {
      return window.withCsrfHeaders(method);
    }
    const m = (method || 'GET').toUpperCase();
    const headers = { 'Content-Type': 'application/json' };
    if (m === 'POST' || m === 'PUT' || m === 'DELETE' || m === 'PATCH') {
      const cookie = (document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/) || [])[1] || '';
      headers['X-CSRF-Token'] = decodeURIComponent(cookie);
    }
    return headers;
  }

  // ---- tiny render helpers
  function chip(status, label) {
    const cls = status === 'pass' ? 'chip-pass'
              : status === 'fail' ? 'chip-fail'
              : status === 'warn' ? 'chip-warn'
              : 'chip-pending';
    return `<span class="chip ${cls}">${label || status.toUpperCase()}</span>`;
  }

  function recordResult(sectionId, entry) {
    if (!RESULTS.sections[sectionId]) RESULTS.sections[sectionId] = [];
    RESULTS.sections[sectionId].push(entry);
  }

  // ---- HTML escape for safe interpolation into innerHTML
  function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ---- copy-to-clipboard helper that also flashes a temporary label
  function attachCopyBtn(btn, getText) {
    btn.addEventListener('click', function () {
      const txt = typeof getText === 'function' ? getText() : getText;
      if (!navigator.clipboard) {
        alert('Clipboard unavailable; select the <pre> text manually.');
        return;
      }
      navigator.clipboard.writeText(txt).then(
        () => {
          const orig = btn.textContent;
          btn.textContent = 'Copied!';
          setTimeout(() => { btn.textContent = orig; }, 1200);
        },
        (e) => alert('Copy failed: ' + (e && e.message ? e.message : e))
      );
    });
  }

  // ---- run a single test via POST /api/test-runner/run
  async function runTest(testId) {
    const resp = await fetch('/api/test-runner/run', {
      method: 'POST',
      headers: csrfHeaders('POST'),
      body: JSON.stringify({ test_id: testId }),
    });
    if (!resp.ok) {
      let err = `HTTP ${resp.status}`;
      try { const j = await resp.json(); err = j.error || err; } catch (_) {}
      throw new Error(err);
    }
    return resp.json();
  }

  // ---- decide PASS/FAIL chip from a run result
  function chipForResult(test, result) {
    if (!result || !result.success) return chip('fail', 'ERROR');
    if (result.timed_out) return chip('fail', 'TIMEOUT');
    if (test.kind === 'probe') {
      return (result.exit_code >= 200 && result.exit_code < 300)
        ? chip('pass', `HTTP ${result.exit_code}`)
        : chip('fail', `HTTP ${result.exit_code}`);
    }
    return result.exit_code === 0 ? chip('pass') : chip('fail', `exit ${result.exit_code}`);
  }

  // ---- render a single test card with a Run button + output area
  function renderTestCard(test) {
    const card = document.createElement('div');
    card.className = 'tr-card';
    card.dataset.testId = test.id;
    card.innerHTML = `
      <div class="tr-card-title">${esc(test.title)}</div>
      <div class="tr-row">
        <button type="button" class="btn btn-sm btn-primary" data-action="run">
          <i class="fas fa-play"></i> Run
        </button>
        <button type="button" class="btn btn-sm btn-secondary tr-copy-btn"
                data-action="copy" style="display:none;">Copy</button>
        <span class="tr-status">${chip('pending')}</span>
        <span class="tr-meta" style="color:var(--text-secondary,#94a3b8);font-size:0.75rem;"></span>
      </div>
      <div class="tr-expected">${esc(test.description || '')}<br>
        <code style="font-size:0.7rem;">${esc(test.cmd_preview || '')}</code>
      </div>
      <pre class="tr-output" style="display:none;"></pre>
    `;
    const runBtn = card.querySelector('[data-action="run"]');
    const copyBtn = card.querySelector('[data-action="copy"]');
    const statusEl = card.querySelector('.tr-status');
    const metaEl = card.querySelector('.tr-meta');
    const outEl = card.querySelector('.tr-output');

    runBtn.addEventListener('click', async () => {
      runBtn.disabled = true;
      statusEl.innerHTML = chip('pending', 'RUNNING…');
      outEl.style.display = 'none';
      copyBtn.style.display = 'none';
      try {
        const result = await runTest(test.id);
        statusEl.innerHTML = chipForResult(test, result);
        metaEl.textContent = `${result.duration_ms || 0} ms`;
        const stdout = result.stdout || '';
        const stderr = result.stderr || '';
        const body = stderr
          ? `--- stdout ---\n${stdout || '(empty)'}\n\n--- stderr ---\n${stderr}`
          : (stdout || '(no output)');
        outEl.textContent = body;
        outEl.style.display = 'block';
        copyBtn.style.display = 'inline-block';
        attachCopyBtn(copyBtn, body);
        recordResult(test.category, {
          id: test.id, title: test.title, chip: statusEl.textContent.trim(),
          duration_ms: result.duration_ms, body: body,
        });
      } catch (e) {
        statusEl.innerHTML = chip('fail', 'ERROR');
        outEl.textContent = 'Run failed: ' + (e && e.message ? e.message : e);
        outEl.style.display = 'block';
      } finally {
        runBtn.disabled = false;
      }
    });
    return card;
  }

  // ---- fetch catalog and populate Sections B/C/D
  async function loadCatalog() {
    let catalog;
    try {
      const r = await fetch('/api/test-runner/tests');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      catalog = (data && data.tests) || [];
    } catch (e) {
      console.error('Test catalog fetch failed:', e);
      return;
    }
    const targets = {
      scripts: document.getElementById('tr-section-b-body'),
      api:     document.getElementById('tr-section-c-body'),
      db:      document.getElementById('tr-section-d-body'),
    };
    // Clear any pending chip in the section headers
    ['b','c','d'].forEach(s => {
      const sec = document.getElementById('tr-section-' + s);
      if (sec) sec.querySelectorAll('.chip-pending').forEach(c => c.remove());
    });
    Object.values(targets).forEach(t => {
      if (t) {
        t.innerHTML = '';
        t.classList.add('tr-grid');
      }
    });
    catalog.forEach(test => {
      const target = targets[test.category];
      if (target) target.appendChild(renderTestCard(test));
    });
  }

  // ---- DOM ready: kick off catalog load + wire Copy All
  document.addEventListener('DOMContentLoaded', function () {
    loadCatalog();
    const copyBtn = document.getElementById('tr-copy-all');
    if (copyBtn) {
      copyBtn.addEventListener('click', copyAllResults);
    }
  });

  // ---- "Copy All Results" — assemble a single markdown blob across
  //      every section so the operator pastes one block back to chat.
  function copyAllResults() {
    const lines = ['# Test Runner — results', ''];
    const order = ['ui_clones', 'scripts', 'api', 'db'];
    const titles = {
      ui_clones: 'A. UI Verification Clones',
      scripts:   'B. Scripts',
      api:       'C. API Probes',
      db:        'D. DB Probes',
    };
    let any = false;
    order.forEach(sec => {
      const entries = RESULTS.sections[sec] || [];
      if (!entries.length) return;
      any = true;
      lines.push('## ' + titles[sec]);
      lines.push('');
      entries.forEach(e => {
        lines.push('### ' + e.title + ' — ' + (e.chip || '?'));
        if (e.duration_ms !== undefined) lines.push('_' + e.duration_ms + ' ms_');
        if (e.body) {
          lines.push('```');
          lines.push(e.body);
          lines.push('```');
        }
        lines.push('');
      });
    });
    if (!any) {
      lines.push('_(no tests have been run yet — click each section\'s Run buttons first)_');
    }
    const md = lines.join('\n');
    if (!navigator.clipboard) {
      alert('Clipboard unavailable; here it is:\n\n' + md);
      return;
    }
    navigator.clipboard.writeText(md).then(
      () => alert('Copied ' + md.length + ' chars to clipboard. Paste back in chat.'),
      (e) => alert('Copy failed: ' + (e && e.message ? e.message : e))
    );
  }

  // ---- expose for later commits
  window.TR = {
    csrfHeaders: csrfHeaders,
    chip: chip,
    recordResult: recordResult,
    results: RESULTS,
  };
})();
