"""
Profile Embedding Generation
Benchmarks the embedding service with different batch sizes
"""

import sys
import time
import random
import string
import statistics
from typing import List

# Add parent directory to path
sys.path.append(".")

from backend.core.embeddings import EmbeddingService
from backend.utils.logger import setup_logging

def generate_random_text(min_len=50, max_len=200):
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + string.digits + " ", k=length))

def run_benchmark():
    setup_logging()
    
    print("🚀 Starting Embedding Service Benchmark")
    print("-" * 50)
    
    # Initialize service
    print("Initializing service...")
    try:
        service = EmbeddingService()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    # Generate dataset
    num_texts = 100
    print(f"Generating {num_texts} sample texts...")
    texts = [generate_random_text() for _ in range(num_texts)]
    
    # Test cases
    batch_sizes = [1, 8, 32, 64]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting Batch Size: {batch_size}")
        
        times = []
        # Warmup
        service.embed_batch(texts[:min(batch_size, len(texts))], batch_size=batch_size)
        
        # Run iterations
        iterations = 5
        for i in range(iterations):
            start = time.time()
            service.embed_batch(texts, batch_size=batch_size)
            end = time.time()
            duration = end - start
            times.append(duration)
            print(f"  Iteration {i+1}: {duration:.4f}s")
            
        avg_time = statistics.mean(times)
        texts_per_sec = num_texts / avg_time
        results[batch_size] = texts_per_sec
        print(f"  Average: {avg_time:.4f}s")
        print(f"  Throughput: {texts_per_sec:.2f} texts/sec")

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"{'Batch Size':<15} | {'Throughput (texts/s)':<20} | {'Speedup'}")
    print("-" * 55)
    
    base_throughput = results[1]
    
    for bs in batch_sizes:
        throughput = results[bs]
        speedup = throughput / base_throughput
        print(f"{bs:<15} | {throughput:<20.2f} | {speedup:.2fx}")
        
    print("\n✅ Benchmark Complete")

if __name__ == "__main__":
    run_benchmark()
