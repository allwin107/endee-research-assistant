"""
Profile Memory Usage
Script to stress test memory usage and verify optimizer
"""

import sys
import time
import os
import psutil
from typing import List

# Add parent path
sys.path.append(".")

from backend.utils.memory import MemoryOptimizer, get_memory_optimizer
from backend.utils.logger import setup_logging

def create_large_list(size: int) -> List[int]:
    """Create a large list to consume memory"""
    print(f"Allocating list with {size} integers...")
    return [i for i in range(size)]

def run_memory_profile():
    setup_logging()
    
    print("🧠 Starting Memory Profiler")
    print("-" * 50)
    
    optimizer = get_memory_optimizer()
    # Set low threshold for testing
    optimizer.threshold_mb = 100  
    
    print(f"Initial Memory: {optimizer.get_current_usage():.2f} MB")
    
    # 1. Allocation Test
    data = []
    try:
        # Allocate ~200MB (approx 25M integers is ~200MB+ in Python list overhead)
        # Python ints are 28 bytes, plus list overhead.
        # 5,000,000 ints * 28 bytes ≈ 140MB
        chunk_size = 1_000_000
        for i in range(10):
            print(f"Allocating chunk {i+1}...")
            data.append(create_large_list(chunk_size))
            usage = optimizer.get_current_usage()
            print(f"Current Usage: {usage:.2f} MB")
            
            # Manually trigger check (since thread runs every 60s)
            if usage > optimizer.threshold_mb:
                print(">> Triggering Optimization...")
                optimizer.optimize()
                
            time.sleep(0.5)
            
    except MemoryError:
        print("!! Out of Memory !!")
        
    print("-" * 30)
    print("Releasing reference to data...")
    data = None
    
    print("Forcing Optimization...")
    optimizer.optimize()
    print(f"Final Memory: {optimizer.get_current_usage():.2f} MB")
    
    print("\n✅ profiling complete")

if __name__ == "__main__":
    run_memory_profile()
