/**
 * timezone.js — shared timestamp helpers.
 *
 * The dashboard server stores all timestamps in UTC but emits them as
 * NAIVE ISO strings (no trailing Z and no +HH:MM offset), e.g.
 *   "2026-05-19T19:33:52"
 *
 * JavaScript's `new Date("2026-05-19T19:33:52")` interprets a naive
 * string as LOCAL time, which shifts the displayed time by the
 * operator's UTC offset. For an operator at UTC+3 a fresh row from
 * "now" appears 3 hours in the future and "relative" widgets read
 * "in 3h" / "3h ago" depending on direction.
 *
 * The fix shipped in orchestrator.html: append 'Z' when no offset is
 * present so the string is parsed as UTC, then render with
 * `toLocaleString()` which respects the browser's timezone.
 *
 * This module exposes the helpers globally on `window` so any
 * template (Copy Trading, DEX, Sniper, AI, etc.) can call them
 * without an import.
 *
 *   window.parseUtcTimestamp(iso)    -> Date | null
 *   window.formatLocalDateTime(iso)  -> string ("5/20/2026, 10:57:00 PM")
 *   window.formatTimeAgo(iso)        -> string ("3m ago", "2h ago", local datetime)
 *
 * Pass an ISO string OR an existing Date OR null/undefined.
 */
(function (global) {
  'use strict';

  function _hasOffset(s) {
    return /Z|[+-]\d{2}:?\d{2}$/.test(s);
  }

  /**
   * Parse a server-emitted timestamp into a Date.
   * If the input is naive (no Z / no offset), it is treated as UTC.
   * Accepts Date objects (returned as-is) and falsy values (returns null).
   */
  function parseUtcTimestamp(value) {
    if (value === null || value === undefined || value === '') return null;
    if (value instanceof Date) {
      return isNaN(value.getTime()) ? null : value;
    }
    try {
      let s = String(value);
      if (!_hasOffset(s)) s += 'Z';
      const d = new Date(s);
      return isNaN(d.getTime()) ? null : d;
    } catch (_) {
      return null;
    }
  }

  /**
   * Render an ISO timestamp in the operator's local timezone using
   * toLocaleString(). Returns an empty string if the value is null
   * or unparseable.
   */
  function formatLocalDateTime(value, options) {
    const d = parseUtcTimestamp(value);
    if (!d) return '';
    try {
      return options ? d.toLocaleString(undefined, options) : d.toLocaleString();
    } catch (_) {
      return d.toLocaleString();
    }
  }

  /**
   * Render a relative-time string ("just now", "5m ago", "2h ago")
   * with a fallback to the absolute local datetime for entries older
   * than 24 hours. Mirrors the orchestrator.html formatTime() shape.
   */
  function formatTimeAgo(value) {
    const d = parseUtcTimestamp(value);
    if (!d) return '';
    const diffMs = Date.now() - d.getTime();
    if (diffMs < 0) {
      // Future timestamp (clock skew / scheduled item) — show absolute.
      return d.toLocaleString();
    }
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return diffMin + 'm ago';
    const hr = Math.floor(diffMin / 60);
    if (hr < 24) return hr + 'h ago';
    return d.toLocaleString();
  }

  global.parseUtcTimestamp = parseUtcTimestamp;
  global.formatLocalDateTime = formatLocalDateTime;
  global.formatTimeAgo = formatTimeAgo;
})(typeof window !== 'undefined' ? window : globalThis);
