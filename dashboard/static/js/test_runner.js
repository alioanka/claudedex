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

  // ---- DOM ready: stub init — sections A/B/C/D are filled in by later commits
  document.addEventListener('DOMContentLoaded', function () {
    // Wire up the global Copy All Results button (built out in commit 5).
    const copyBtn = document.getElementById('tr-copy-all');
    if (copyBtn) {
      copyBtn.addEventListener('click', function () {
        const md = '# Test Runner — pending sections will populate as they ship.\n' +
                   '_(Sections B/C/D require Agent 3 backend routes.)_';
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(md).then(
            () => alert('Copied placeholder summary — full implementation lands in commit 5.'),
            (e) => alert('Copy failed: ' + (e && e.message ? e.message : e))
          );
        }
      });
    }
  });

  // ---- expose for later commits
  window.TR = {
    csrfHeaders: csrfHeaders,
    chip: chip,
    recordResult: recordResult,
    results: RESULTS,
  };
})();
