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

  // ---- render the small coloured tag pills next to a test title.
  //      Tags come from /api/test-runner/tests (see _compute_tags on the
  //      backend). Allowed values are pinned to the CSS classes in
  //      test_runner.html so a stray tag silently renders as nothing.
  const ALLOWED_TAG_CLASSES = {
    'must': 'tag-must',
    'new': 'tag-new',
    'p0': 'tag-p0',
    'flaky': 'tag-flaky',
    'expected-empty': 'tag-expected-empty',
  };
  function renderTagPills(tags) {
    if (!Array.isArray(tags) || !tags.length) return '';
    const pills = tags
      .filter(t => ALLOWED_TAG_CLASSES[t])
      .map(t => `<span class="tag-pill ${ALLOWED_TAG_CLASSES[t]}" title="tag: ${esc(t)}">${esc(t.toUpperCase())}</span>`)
      .join('');
    return pills ? `<span class="tr-tag-row">${pills}</span>` : '';
  }

  // ---- safe wrapper around the recently-run history push. No-ops
  //      if the toolbar hasn't loaded yet (e.g. Section A clones fire
  //      before initToolbar wires window.TR_pushRecent).
  function pushRecentSafe(test, status) {
    if (typeof window.TR_pushRecent !== 'function') return;
    window.TR_pushRecent(test.category, {
      id: test.id, title: test.title, status: status, ts: Date.now(),
    });
  }

  // ---- copy-to-clipboard helper. Falls back to a textarea-modal on
  //      http:// remote URLs where navigator.clipboard is blocked.
  function attachCopyBtn(btn, getText) {
    btn.addEventListener('click', function () {
      const txt = typeof getText === 'function' ? getText() : getText;
      const flash = () => {
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1200);
      };
      const tryAsync = navigator.clipboard && navigator.clipboard.writeText;
      if (tryAsync) {
        navigator.clipboard.writeText(txt).then(
          flash,
          () => showCopyFallback(txt)
        );
      } else {
        showCopyFallback(txt);
      }
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
    // Stash tags + searchable text on the card so client-side filter +
    // search chips (commits 3-4) can show/hide without a DOM walk.
    const tags = Array.isArray(test.tags) ? test.tags : [];
    card.dataset.tags = tags.join(',');
    card.dataset.searchBlob = ((test.title || '') + ' ' + (test.description || '')).toLowerCase();
    card.innerHTML = `
      <div class="tr-card-title">${esc(test.title)}${renderTagPills(tags)}</div>
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
        // Push into the per-section recently-run history strip. Pass
        // judged by exit_code (0 for bash/db_query, 2xx for probe).
        const passed = result && result.success && !result.timed_out && (
          test.kind === 'probe'
            ? (result.exit_code >= 200 && result.exit_code < 300)
            : (result.exit_code === 0)
        );
        pushRecentSafe(test, passed ? 'pass' : 'fail');
      } catch (e) {
        statusEl.innerHTML = chip('fail', 'ERROR');
        outEl.textContent = 'Run failed: ' + (e && e.message ? e.message : e);
        outEl.style.display = 'block';
        pushRecentSafe(test, 'fail');
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
    // Populate filter-chip totals (count of every tagged entry, ignoring
    // current filters) so the operator sees "Must 16" not "Must 0".
    const totals = {};
    catalog.forEach(t => (t.tags || []).forEach(tag => {
      totals[tag] = (totals[tag] || 0) + 1;
    }));
    document.querySelectorAll('[data-tag-count]').forEach(el => {
      const tag = el.dataset.tagCount;
      el.textContent = totals[tag] || 0;
    });
    // First paint: count chips reflect every card, recent-history strips
    // hydrate from localStorage, and the search/chip listeners arm.
    applyFilters();
    refreshAllRecentStrips();
  }

  // ─── Toolbar wiring: search box + filter chips + Run Filtered ────
  // State is held in two sets: active tag filters (`STATE.tags`) and a
  // lowercased search query (`STATE.q`). Both update on every event;
  // applyFilters walks every .tr-card once and toggles .tr-hidden,
  // then refreshes the per-section count chips.
  const STATE = { tags: new Set(), q: '' };

  function applyFilters() {
    const q = STATE.q.trim();
    const wantTags = STATE.tags;
    const perSection = { scripts: 0, api: 0, db: 0 };
    document.querySelectorAll('.tr-card[data-test-id]').forEach(card => {
      const cardTags = (card.dataset.tags || '').split(',').filter(Boolean);
      // Tag check: card must have AT LEAST ONE of the active tags (OR).
      let tagOk = wantTags.size === 0;
      if (!tagOk) {
        for (const t of cardTags) {
          if (wantTags.has(t)) { tagOk = true; break; }
        }
      }
      const searchOk = !q || (card.dataset.searchBlob || '').indexOf(q) !== -1;
      const visible = tagOk && searchOk;
      card.classList.toggle('tr-hidden', !visible);
      if (visible) {
        // Determine which section the card lives in by walking up to the
        // nearest [data-section] container.
        const sec = card.closest('[data-section]');
        if (sec) {
          const k = sec.dataset.section;
          if (perSection[k] !== undefined) perSection[k]++;
        }
      }
    });
    // Live-update per-section count chips.
    document.querySelectorAll('[data-section-count]').forEach(el => {
      const k = el.dataset.sectionCount;
      el.textContent = perSection[k] || 0;
    });
  }

  function initToolbar() {
    const search = document.getElementById('tr-search');
    if (search) {
      search.addEventListener('input', (e) => {
        STATE.q = (e.target.value || '').toLowerCase();
        applyFilters();
      });
    }
    document.querySelectorAll('.tr-filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const f = chip.dataset.filter;
        if (f === '__all') {
          STATE.tags.clear();
          document.querySelectorAll('.tr-filter-chip').forEach(c =>
            c.classList.toggle('active', c.dataset.filter === '__all'));
        } else {
          if (STATE.tags.has(f)) STATE.tags.delete(f);
          else STATE.tags.add(f);
          chip.classList.toggle('active', STATE.tags.has(f));
          // Hide "All" highlight when any tag filter is active.
          const allChip = document.querySelector('.tr-filter-chip[data-filter="__all"]');
          if (allChip) allChip.classList.toggle('active', STATE.tags.size === 0);
        }
        applyFilters();
      });
    });
    // Pre-light the All chip so the operator sees the initial state.
    const allChip = document.querySelector('.tr-filter-chip[data-filter="__all"]');
    if (allChip) allChip.classList.add('active');

    const runBtn = document.getElementById('tr-run-filtered');
    if (runBtn) runBtn.addEventListener('click', runFiltered);
  }

  // ---- "Run Filtered" — sequentially click the Run button on every
  //      currently-visible card. Sequential (not parallel) because some
  //      tests touch the same DB rows and we want the operator to be
  //      able to read the progress meter without a 121-way race.
  async function runFiltered() {
    const cards = Array.from(document.querySelectorAll(
      '.tr-card[data-test-id]:not(.tr-hidden)'));
    if (!cards.length) {
      alert('No visible tests — clear a filter or widen the search.');
      return;
    }
    const progress = document.getElementById('tr-progress');
    const runBtn = document.getElementById('tr-run-filtered');
    if (runBtn) runBtn.disabled = true;
    let done = 0;
    for (const card of cards) {
      done++;
      if (progress) progress.textContent = `${done}/${cards.length} running…`;
      const btn = card.querySelector('[data-action="run"]');
      if (!btn) continue;
      // Re-use the existing per-card click handler so results record
      // into the Copy-All blob just like a manual click.
      btn.click();
      // Wait until the per-card button re-enables, polling every 150ms.
      // Hard timeout (180s) so a stuck network call doesn't freeze the
      // whole sweep — we surface and move on.
      const t0 = Date.now();
      while (btn.disabled && (Date.now() - t0) < 180000) {
        await new Promise(r => setTimeout(r, 150));
      }
    }
    if (progress) progress.textContent = `${done}/${cards.length} complete`;
    if (runBtn) runBtn.disabled = false;
  }

  // ─── Recently-run history (last 5 per section, in localStorage) ───
  // Key shape: `tr_recent_<section>` → JSON array of {id, title,
  // status:'pass'|'fail', ts}. We touch this from inside the per-card
  // Run handler (renderTestCard) by exposing pushRecent on window.TR.
  const RECENT_LIMIT = 5;
  function recentKey(section) { return 'tr_recent_' + section; }
  function loadRecent(section) {
    try { return JSON.parse(localStorage.getItem(recentKey(section)) || '[]'); }
    catch (_) { return []; }
  }
  function pushRecent(section, item) {
    const list = loadRecent(section);
    list.unshift(item);
    while (list.length > RECENT_LIMIT) list.pop();
    try { localStorage.setItem(recentKey(section), JSON.stringify(list)); }
    catch (_) {}
    refreshRecentStrip(section);
  }
  function ageString(ts) {
    const s = Math.max(0, Math.floor((Date.now() - ts) / 1000));
    if (s < 60) return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return Math.floor(s / 86400) + 'd ago';
  }
  function refreshRecentStrip(section) {
    const host = document.querySelector(`[data-section-recent="${section}"]`);
    if (!host) return;
    const list = loadRecent(section);
    if (!list.length) { host.innerHTML = ''; return; }
    const parts = ['<span class="tr-recent-label">Recent:</span>'];
    list.forEach(item => {
      const dot = item.status === 'pass' ? 'pass' : 'fail';
      const t = esc((item.title || item.id || '').slice(0, 40));
      parts.push(
        `<span class="tr-recent-pill" data-jump-to="${esc(item.id)}" ` +
        `title="${esc(item.title || item.id)} — click to jump">` +
        `<span class="dot ${dot}"></span>${t}` +
        `<span class="age">${ageString(item.ts)}</span></span>`
      );
    });
    host.innerHTML = parts.join('');
    host.querySelectorAll('[data-jump-to]').forEach(p => {
      p.addEventListener('click', () => {
        const id = p.dataset.jumpTo;
        const card = document.querySelector(`.tr-card[data-test-id="${id}"]`);
        if (card) { card.scrollIntoView({behavior:'smooth', block:'center'}); card.style.outline='2px solid #3b82f6'; setTimeout(()=>card.style.outline='',1500); }
      });
    });
  }
  function refreshAllRecentStrips() {
    ['scripts','api','db'].forEach(refreshRecentStrip);
  }
  window.TR_pushRecent = pushRecent; // exposed for renderTestCard hook

  // ─── Section A clones — each verify() reads the same data the real
  //     page uses, sets the inline preview, and reports PASS/FAIL.

  async function fetchJson(url) {
    const r = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }

  function setCardStatus(cardId, status, label, body) {
    const card = document.getElementById(cardId);
    if (!card) return;
    const statusEl = card.querySelector('.tr-status');
    if (statusEl) statusEl.innerHTML = chip(status, label);
    if (body !== undefined) {
      const outEl = card.querySelector('.tr-output');
      if (outEl) {
        outEl.textContent = body;
        outEl.style.display = 'block';
      }
    }
    // Record into the Copy All Results blob.
    const title = card.querySelector('.tr-card-title');
    recordResult('ui_clones', {
      id: cardId,
      title: title ? title.textContent : cardId,
      chip: (status || '?').toUpperCase() + (label ? ' (' + label + ')' : ''),
      body: body || '',
    });
  }

  // A1: MODE badge clone — mirrors refreshBotModeBadge() in main.js
  // but writes into the cloned div ids (tr-mode-*).
  async function verifyModeBadge() {
    const badge = document.getElementById('tr-mode-badge');
    const txt = document.getElementById('tr-mode-text');
    if (!badge || !txt) return;
    try {
      const payload = await fetchJson('/api/bot/status');
      const data = (payload && payload.data) ? payload.data : payload;
      badge.classList.remove('bot-mode-live', 'bot-mode-dry', 'bot-mode-unknown');
      let pass = false, body = '';
      if (data.dry_run === undefined || data.dry_run === null) {
        badge.classList.add('bot-mode-unknown');
        txt.textContent = 'UNKNOWN';
        body = JSON.stringify(data, null, 2);
      } else if (data.dry_run === false) {
        badge.classList.add('bot-mode-live');
        txt.textContent = '🔴 LIVE TRADING';
        pass = true;
        body = 'dry_run=false → 🔴 LIVE TRADING\n' + JSON.stringify(data, null, 2);
      } else {
        badge.classList.add('bot-mode-dry');
        txt.textContent = '🔵 DRY-RUN';
        pass = true;
        body = 'dry_run=true → 🔵 DRY-RUN\n' + JSON.stringify(data, null, 2);
      }
      setCardStatus('tr-clone-mode-badge', pass ? 'pass' : 'fail',
                    pass ? null : 'UNKNOWN', body);
    } catch (e) {
      badge.classList.add('bot-mode-unknown');
      txt.textContent = 'STATUS UNAVAILABLE';
      setCardStatus('tr-clone-mode-badge', 'fail', 'fetch error',
                    'error: ' + (e && e.message ? e.message : e));
    }
  }

  // A2: Sniper cap tile clone — mirrors dashboard_sniper.html cap-warn
  // logic from lines 625-665.
  async function verifySniperCap() {
    const el = document.getElementById('tr-sniper-cap');
    if (!el) return;
    try {
      const data = await fetchJson('/api/sniper/stats');
      const active = (data.active_positions_effective != null)
        ? data.active_positions_effective
        : (data.active_positions || 0);
      const cap = data.max_active_positions || 0;
      let text = 'Active: ' + active;
      let status = 'pass', label = null;
      if (cap > 0) {
        const pct = (active / cap) * 100;
        if (pct >= 95) {
          text = `Active: ${active}/${cap} ⛔ CAPPED — new snipes blocked`;
          label = 'CAPPED';
        } else if (pct >= 80) {
          text = `Active: ${active}/${cap} ⚠ near cap`;
          label = 'near cap';
        } else {
          text = `Active: ${active}/${cap}`;
        }
      } else {
        // Cap missing — exactly the failure mode the fix addresses.
        text = `Active: ${active} (cap missing!)`;
        status = 'fail';
        label = 'no cap';
      }
      el.textContent = text;
      const body =
        `active_positions          = ${data.active_positions}\n` +
        `active_positions_live     = ${data.active_positions_live}\n` +
        `active_positions_effective= ${data.active_positions_effective}\n` +
        `max_active_positions      = ${data.max_active_positions}\n` +
        `status                    = ${data.status}`;
      setCardStatus('tr-clone-sniper-cap', status, label, body);
    } catch (e) {
      el.textContent = 'Active: error';
      setCardStatus('tr-clone-sniper-cap', 'fail', 'fetch error',
                    'error: ' + (e && e.message ? e.message : e));
    }
  }

  // A3: Risk-critical CSS — operator-judged. Two buttons: "Mark
  // legible" (pass) and "Mark unreadable" (fail). No auto-verify
  // because contrast is subjective.
  function wireRiskCss() {
    const card = document.getElementById('tr-clone-risk-css');
    if (!card) return;
    const passBtn = card.querySelector('[data-action="verify"]');
    const failBtn = card.querySelector('[data-action="fail"]');
    if (passBtn) passBtn.addEventListener('click', () => {
      setCardStatus('tr-clone-risk-css', 'pass', 'legible',
        'Operator confirmed risk-critical inputs are readable.');
    });
    if (failBtn) failBtn.addEventListener('click', () => {
      setCardStatus('tr-clone-risk-css', 'fail', 'unreadable',
        'Operator flagged risk-critical inputs as unreadable.');
    });
  }

  // A4: Module Overview — fetch /api/modules and confirm each
  // module's `enabled` flag matches the env-flag source-of-truth.
  // Previously this hard-coded EXPECTED_ENABLED = ['sniper',
  // 'arbitrage'] from the operator's earlier deployment, which now
  // FAILS the moment they enable more modules. The check is now
  // self-consistent: each module reports its own enabled state from
  // the env flag, so we just verify the response shape is sane
  // (every module has enabled/status/effective_dry_run fields).

  async function verifyModuleOverview() {
    const grid = document.getElementById('tr-module-grid');
    if (!grid) return;
    grid.innerHTML = '<span style="color:var(--text-secondary,#94a3b8);">loading…</span>';
    try {
      const payload = await fetchJson('/api/modules');
      const data = (payload && payload.data) ? payload.data : payload;
      // /api/modules can return either an array or {modules: {...}}.
      let entries = [];
      if (Array.isArray(data)) {
        entries = data.map(m => [m.name || m.key, m]);
      } else if (data && data.modules) {
        entries = Object.entries(data.modules);
      }
      grid.innerHTML = '';
      const mismatches = [];
      const rows = [];
      entries.forEach(([name, mod]) => {
        const enabled = !!mod.enabled;
        const status = String(mod.status || (enabled ? 'ENABLED' : 'DISABLED'));
        // Self-consistency checks (instead of comparing against a
        // hardcoded expectation):
        //   - if enabled=true, status must NOT start with DISABLED
        //   - if enabled=false, status MUST be DISABLED
        //   - effective_dry_run field must be present (boolean)
        let match = true;
        if (enabled && status.startsWith('DISABLED')) {
          match = false;
          mismatches.push(`${name}: enabled=true but status=DISABLED`);
        }
        if (!enabled && !status.startsWith('DISABLED')) {
          match = false;
          mismatches.push(`${name}: enabled=false but status=${status}`);
        }
        if (typeof mod.effective_dry_run !== 'boolean') {
          match = false;
          mismatches.push(`${name}: missing effective_dry_run field`);
        }
        const cls = enabled ? 'chip-pass' : 'chip-pending';
        const dryChip = (mod.effective_dry_run === false)
          ? '<span class="chip chip-fail" style="margin-left:4px;">LIVE</span>'
          : (mod.effective_dry_run === true ? '<span class="chip" style="background:#3b82f6;color:#fff;margin-left:4px;">DRY</span>' : '');
        const div = document.createElement('div');
        div.style.cssText = 'padding:6px 8px;border:1px solid var(--border-color,#334155);' +
                            'border-radius:6px;background:var(--bg-secondary,#1e293b);';
        div.innerHTML =
          `<div style="font-weight:600;font-size:0.85rem;">${esc(name)}</div>` +
          `<div style="margin:4px 0 0 0;"><span class="chip ${cls}">${esc(status)}</span>${dryChip}</div>` +
          (match ? '' :
            `<div style="color:#ef4444;font-size:0.7rem;margin-top:4px;">⚠ shape inconsistency</div>`);
        grid.appendChild(div);
        rows.push(`${name.padEnd(20)} status=${status.padEnd(20)} enabled=${enabled} dry=${mod.effective_dry_run}`);
      });
      const body = rows.join('\n') + (mismatches.length
        ? '\n\nINCONSISTENCIES:\n' + mismatches.map(s => '  ' + s).join('\n') : '');
      setCardStatus('tr-clone-module-overview',
        mismatches.length === 0 ? 'pass' : 'fail',
        mismatches.length ? `${mismatches.length} issue` : null,
        body);
    } catch (e) {
      grid.innerHTML = '<span style="color:#ef4444;">fetch error</span>';
      setCardStatus('tr-clone-module-overview', 'fail', 'fetch error',
        'error: ' + (e && e.message ? e.message : e));
    }
  }

  // A5: Analytics module switcher — sequentially probe each module's
  // /api/analytics/performance endpoint. Pass if every one returns
  // success=true.
  const ANALYTICS_MODULES = [
    'dex_trading', 'futures', 'solana', 'sniper',
    'arbitrage', 'copy_trading', 'ai_analysis',
  ];

  async function verifyAnalyticsSwitcher() {
    const tabsEl = document.getElementById('tr-analytics-tabs');
    if (!tabsEl) return;
    tabsEl.innerHTML = '';
    const lines = [];
    let failCount = 0;
    for (const mod of ANALYTICS_MODULES) {
      const tab = document.createElement('div');
      tab.style.cssText = 'padding:4px 10px;border-radius:6px;font-size:0.75rem;' +
                          'background:var(--bg-secondary,#1e293b);' +
                          'border:1px solid var(--border-color,#334155);';
      tab.textContent = mod + '…';
      tabsEl.appendChild(tab);
      try {
        const url = `/api/analytics/performance/${mod}?timeframe=all`;
        const r = await fetch(url, { headers: { 'Accept': 'application/json' } });
        const ok = r.ok;
        let payload = {};
        try { payload = await r.json(); } catch (_) {}
        const success = ok && (payload.success !== false);
        const trades = (payload.data && payload.data.total_trades) ?? 0;
        tab.textContent = `${mod} ${success ? '✓' : '✗'} (${r.status}, ${trades} tr)`;
        tab.style.background = success ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)';
        tab.style.borderColor = success ? '#10b981' : '#ef4444';
        if (!success) failCount++;
        lines.push(`${mod.padEnd(15)} HTTP ${r.status}  trades=${trades}  success=${success}`);
      } catch (e) {
        failCount++;
        tab.textContent = `${mod} ✗ (error)`;
        tab.style.background = 'rgba(239,68,68,0.2)';
        lines.push(`${mod.padEnd(15)} ERROR: ${e.message || e}`);
      }
    }
    setCardStatus('tr-clone-analytics-switcher',
      failCount === 0 ? 'pass' : (failCount === ANALYTICS_MODULES.length ? 'fail' : 'warn'),
      failCount === 0 ? null : `${failCount}/${ANALYTICS_MODULES.length} fail`,
      lines.join('\n'));
  }

  // A6: CSRF probe — POSTs against /api/test-runner/run with the
  // CSRF header set; failure mode would be 403 "CSRF token missing
  // or invalid". We use test_id=api_health which is harmless.
  async function verifyCsrfProbe() {
    try {
      const r = await fetch('/api/test-runner/run', {
        method: 'POST',
        headers: csrfHeaders('POST'),
        body: JSON.stringify({ test_id: 'api_health' }),
      });
      const body = await r.text();
      if (r.status === 403 && /csrf/i.test(body)) {
        setCardStatus('tr-clone-csrf-probe', 'fail', '403 CSRF',
          `HTTP ${r.status}\n${body}`);
        return;
      }
      if (!r.ok) {
        setCardStatus('tr-clone-csrf-probe', 'warn', `HTTP ${r.status}`,
          `HTTP ${r.status}\n${body.slice(0, 800)}`);
        return;
      }
      setCardStatus('tr-clone-csrf-probe', 'pass', null,
        `CSRF accepted (HTTP 200). Sample body:\n` + body.slice(0, 400));
    } catch (e) {
      setCardStatus('tr-clone-csrf-probe', 'fail', 'fetch error',
        'error: ' + (e && e.message ? e.message : e));
    }
  }

  // Bind Verify buttons inside Section A cards.
  function wireSectionA() {
    const handlers = {
      'tr-clone-mode-badge': verifyModeBadge,
      'tr-clone-sniper-cap': verifySniperCap,
      'tr-clone-module-overview': verifyModuleOverview,
      'tr-clone-analytics-switcher': verifyAnalyticsSwitcher,
      'tr-clone-csrf-probe': verifyCsrfProbe,
    };
    Object.entries(handlers).forEach(([cardId, fn]) => {
      const card = document.getElementById(cardId);
      if (!card) return;
      const btn = card.querySelector('[data-action="verify"]');
      if (btn) btn.addEventListener('click', fn);
      // Run once on load EXCEPT for the analytics switcher (7 sequential
      // HTTP calls) and CSRF probe (fires a real backend run). Operator
      // clicks Verify when ready.
      if (cardId !== 'tr-clone-analytics-switcher' &&
          cardId !== 'tr-clone-csrf-probe') fn();
    });
    wireRiskCss();
  }

  // ---- DOM ready: kick off catalog load + Section A + Copy All +
  //      sticky-toolbar wiring (search/filter chips/Run Filtered).
  document.addEventListener('DOMContentLoaded', function () {
    initToolbar();
    loadCatalog();
    wireSectionA();
    const copyBtn = document.getElementById('tr-copy-all');
    if (copyBtn) {
      copyBtn.addEventListener('click', copyAllResults);
    }
    // Refresh the "Recent" age strings every 30s so "2m ago" doesn't
    // get stuck. Cheap: 3 sections × ≤5 pills.
    setInterval(refreshAllRecentStrips, 30000);
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
    // navigator.clipboard.writeText requires HTTPS or localhost (secure
    // context); on http://<vps-ip>:8080 it throws. Try it first, then
    // fall back to a modal with a pre-selected textarea the operator
    // can Ctrl+C from.
    const tryAsync = navigator.clipboard && navigator.clipboard.writeText;
    if (tryAsync) {
      navigator.clipboard.writeText(md).then(
        () => alert('Copied ' + md.length + ' chars to clipboard. Paste back in chat.'),
        () => showCopyFallback(md)
      );
    } else {
      showCopyFallback(md);
    }
  }

  // Modal with pre-selected textarea. Works on http:// remote IPs
  // where the async clipboard API is blocked. Operator hits Ctrl+C
  // then closes the modal.
  function showCopyFallback(text) {
    const old = document.getElementById('tr-copy-modal');
    if (old) old.remove();
    const wrap = document.createElement('div');
    wrap.id = 'tr-copy-modal';
    wrap.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.75);' +
      'z-index:9999;display:flex;align-items:center;justify-content:center;padding:20px;';
    wrap.innerHTML = `
      <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;
                  padding:16px;max-width:900px;width:100%;color:#f1f5f9;">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:10px;">
          <strong>Test Runner — Copy results</strong>
          <button id="tr-copy-modal-close" class="btn btn-sm btn-secondary">Close</button>
        </div>
        <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:8px;">
          Clipboard API unavailable on http:// remote URLs. Select the text below
          (the textarea is pre-selected) and copy with Ctrl+C / Cmd+C.
        </div>
        <textarea id="tr-copy-modal-textarea" rows="20"
          style="width:100%;background:#0a0f1c;color:#cbd5e1;border:1px solid #334155;
                 border-radius:6px;padding:10px;font-family:ui-monospace,Menlo,monospace;
                 font-size:0.78rem;"></textarea>
      </div>
    `;
    document.body.appendChild(wrap);
    const ta = document.getElementById('tr-copy-modal-textarea');
    ta.value = text;
    ta.focus();
    ta.select();
    document.getElementById('tr-copy-modal-close').addEventListener('click',
      () => wrap.remove());
    // Click backdrop to close
    wrap.addEventListener('click', (e) => { if (e.target === wrap) wrap.remove(); });
  }

  // ---- expose for later commits
  window.TR = {
    csrfHeaders: csrfHeaders,
    chip: chip,
    recordResult: recordResult,
    results: RESULTS,
  };
})();
