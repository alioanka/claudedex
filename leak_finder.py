#!/usr/bin/env python3
"""
Memory Leak Finder
Run this inside the trading-bot container to find what's leaking
"""

import gc
import sys
from collections import Counter
import time

def analyze_memory():
    """Analyze what's consuming memory"""
    gc.collect()
    
    # Count objects by type
    type_counts = Counter()
    large_objects = []
    
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        type_counts[obj_type] += 1
        
        # Find large collections
        if isinstance(obj, (list, tuple)):
            size = len(obj)
            if size > 100:
                large_objects.append((obj_type, size, id(obj)))
        elif isinstance(obj, dict):
            size = len(obj)
            if size > 100:
                large_objects.append((obj_type, size, id(obj)))
    
    return type_counts, large_objects

print("Starting memory leak detection...")
print("Press Ctrl+C to stop\n")

previous_counts = None

try:
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} - {time.strftime('%H:%M:%S')}")
        print('='*60)
        
        type_counts, large_objects = analyze_memory()
        
        # Show top 15 object types
        print("\nTop 15 Object Types:")
        for obj_type, count in type_counts.most_common(15):
            if previous_counts:
                delta = count - previous_counts.get(obj_type, 0)
                if delta > 0:
                    print(f"  {obj_type:30s}: {count:>8,d}  (+{delta:,d}) ⚠️")
                else:
                    print(f"  {obj_type:30s}: {count:>8,d}")
            else:
                print(f"  {obj_type:30s}: {count:>8,d}")
        
        # Show growing objects
        if previous_counts:
            print("\n🚨 RAPIDLY GROWING TYPES:")
            growing = []
            for obj_type, count in type_counts.items():
                prev = previous_counts.get(obj_type, 0)
                delta = count - prev
                if delta > 100:  # Growing by 100+ per iteration
                    growing.append((obj_type, count, delta))
            
            growing.sort(key=lambda x: x[2], reverse=True)
            if growing:
                for obj_type, count, delta in growing[:10]:
                    print(f"  {obj_type:30s}: {count:>8,d}  (+{delta:,d}/sec)")
            else:
                print("  None detected")
        
        # Show large collections
        if large_objects:
            print("\n📦 Large Collections (>100 items):")
            for obj_type, size, obj_id in sorted(large_objects, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {obj_type:30s}: {size:>8,d} items")
        
        previous_counts = type_counts
        time.sleep(3)

except KeyboardInterrupt:
    print("\n\nStopped by user")
    sys.exit(0)