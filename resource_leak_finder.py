#!/usr/bin/env python3
"""
System Resource Leak Finder
Checks for file descriptors, connections, threads, etc.
"""

import os
import psutil
import time
import asyncio

def check_resources():
    """Check system resources"""
    process = psutil.Process()
    
    # Memory
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    
    # File descriptors
    try:
        num_fds = process.num_fds()
    except:
        num_fds = len(process.open_files())
    
    # Threads
    num_threads = process.num_threads()
    
    # Network connections
    try:
        connections = process.connections()
        num_conns = len(connections)
        established = len([c for c in connections if c.status == 'ESTABLISHED'])
        time_wait = len([c for c in connections if c.status == 'TIME_WAIT'])
    except:
        num_conns = established = time_wait = 0
    
    # Open files
    try:
        open_files = process.open_files()
        num_files = len(open_files)
    except:
        num_files = 0
    
    return {
        'memory_mb': mem_mb,
        'file_descriptors': num_fds,
        'threads': num_threads,
        'connections': num_conns,
        'established': established,
        'time_wait': time_wait,
        'open_files': num_files
    }

print("Monitoring system resources...")
print("Press Ctrl+C to stop\n")

previous = None
iteration = 0

try:
    while True:
        iteration += 1
        current = check_resources()
        
        print(f"\n{'='*70}")
        print(f"Iteration {iteration} - {time.strftime('%H:%M:%S')}")
        print('='*70)
        
        for key, value in current.items():
            if previous:
                delta = value - previous[key]
                indicator = "⚠️" if delta > 0 and key != 'memory_mb' else ""
                if delta != 0:
                    print(f"  {key:25s}: {value:>10.1f}  ({delta:+.1f}) {indicator}")
                else:
                    print(f"  {key:25s}: {value:>10.1f}")
            else:
                print(f"  {key:25s}: {value:>10.1f}")
        
        # Alert on critical issues
        if current['file_descriptors'] > 500:
            print("\n🚨 WARNING: High file descriptor count! Possible FD leak!")
        
        if current['connections'] > 100:
            print("\n🚨 WARNING: High connection count! Possible connection leak!")
        
        if current['threads'] > 50:
            print("\n🚨 WARNING: High thread count! Possible thread leak!")
        
        # Show memory growth rate
        if previous:
            mem_delta = current['memory_mb'] - previous['memory_mb']
            if mem_delta > 50:  # More than 50MB in 3 seconds
                print(f"\n🔥 CRITICAL: Memory growing at {mem_delta:.1f} MB per 3 seconds!")
                print(f"   Projected to hit 3GB in {(3000 - current['memory_mb']) / mem_delta * 3:.0f} seconds")
        
        previous = current
        time.sleep(3)

except KeyboardInterrupt:
    print("\n\nStopped by user")