#!/usr/bin/env bash
# scripts/freeze_watchdog.sh
#
# Captures a tiny resource snapshot every 60s to /var/log/freeze-watchdog/.
# If the VPS becomes unresponsive again, the LAST snapshot on disk will
# show what the system looked like seconds before the freeze. Each
# snapshot is appended to today's file so the entire day fits in a few
# hundred KB.
#
# Install as a systemd service so it survives reboots:
#   sudo cp scripts/freeze_watchdog.sh /usr/local/bin/freeze-watchdog
#   sudo chmod +x /usr/local/bin/freeze-watchdog
#   sudo cp scripts/freeze-watchdog.service /etc/systemd/system/
#   sudo systemctl enable --now freeze-watchdog
#
# After a freeze, read the last lines:
#   tail -100 /var/log/freeze-watchdog/$(date +%F).log

set -u
LOGDIR=/var/log/freeze-watchdog
mkdir -p "$LOGDIR"

snapshot() {
    local f="$LOGDIR/$(date +%F).log"
    {
        echo "==== $(date -u +'%Y-%m-%dT%H:%M:%SZ') ===="
        # 1. Memory (the headline)
        echo "-- /proc/meminfo (head) --"
        head -5 /proc/meminfo
        # 2. Load + uptime
        echo "-- uptime --"
        uptime
        # 3. Top 5 memory consumers
        echo "-- top RSS (5) --"
        ps -eo pid,rss,vsz,%cpu,comm --sort=-rss | head -6
        # 4. Disk usage (fast — no full du)
        echo "-- df / --"
        df -h / | tail -1
        # 5. Open file descriptors (tcp + total)
        echo "-- sockets/fds --"
        echo "tcp_sockets=$(ss -tan 2>/dev/null | wc -l)"
        echo "open_files=$(lsof 2>/dev/null | wc -l || echo 'lsof_unavailable')"
        # 6. conntrack
        echo "conntrack=$(cat /proc/sys/net/netfilter/nf_conntrack_count 2>/dev/null)"
        # 7. Docker (catch the moment a container starts thrashing)
        echo "-- docker stats --"
        timeout 5 docker stats --no-stream \
            --format '{{.Name}}: mem={{.MemUsage}} cpu={{.CPUPerc}}' 2>/dev/null \
            || echo "docker unresponsive"
        # 8. Last 3 kernel messages — show any incipient OOM/IO warning
        echo "-- dmesg tail --"
        dmesg --ctime 2>/dev/null | tail -3 || echo "dmesg unavailable"
        echo ""
    } >> "$f" 2>&1
}

# Keep last 14 days only — auto-rotation.
find "$LOGDIR" -name '*.log' -mtime +14 -delete 2>/dev/null

while true; do
    snapshot
    sleep 60
done
