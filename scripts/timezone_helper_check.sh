#!/usr/bin/env bash
# Dashboard timezone-helper presence check (catalog:
# script_timezone_helper_present; W6 commit 2500f5f).
#
# The dashboard ships a shared client-side helper at
# /static/js/timezone.js that exposes window.parseUtcTimestamp /
# formatLocalDateTime / formatTimeAgo. Server emits naive UTC ISO
# strings; without these helpers, browser JS parses them as LOCAL
# time and shifts every timestamp display by the operator's UTC
# offset (UTC+3 -> "3h ago" on fresh rows). base.html MUST load the
# script before any page-specific JS.
#
# This probe asserts:
#   1. /static/js/timezone.js exists on disk.
#   2. The file exports all three globals via window.<name> = ...
#   3. base.html includes the <script> tag for timezone.js.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

helper=dashboard/static/js/timezone.js
base=dashboard/templates/base.html

if [ ! -f "$helper" ]; then
  echo "FAIL — $helper not found"
  exit 1
fi

if [ ! -f "$base" ]; then
  echo "FAIL — $base not found"
  exit 1
fi

missing=()
for sym in 'window.parseUtcTimestamp' 'window.formatLocalDateTime' 'window.formatTimeAgo'; do
  if ! grep -F "$sym" "$helper" >/dev/null; then
    missing+=("export missing in helper: $sym")
  fi
done

# base.html must reference the script. Allow either ?v={{ ... }} or
# a bare include — both are valid.
if ! grep -F '/static/js/timezone.js' "$base" >/dev/null; then
  missing+=("base.html does not load /static/js/timezone.js")
fi

if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — timezone helper integration broken:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  exit 1
fi

echo "PASS — timezone.js exports parseUtcTimestamp/formatLocalDateTime/formatTimeAgo + loaded in base.html"
