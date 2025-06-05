#!/usr/bin/env python3
"""
Test CPU mining performance
"""

import hashlib
import time
import multiprocessing

def benchmark_single_core():
    """Benchmark single core performance"""
    print("Testing single core performance...")
    
    test_string = "brvdgx5w3p00000001491c"
    hashes = 0
    start = time.time()
    
    while time.time() - start < 5:  # 5 second test
        for i in range(10000):
            msg = test_string + str(hashes + i)
            hashlib.sha256(msg.encode()).hexdigest()
        hashes += 10000
    
    elapsed = time.time() - start
    rate = hashes / elapsed
    print(f"Single core: {rate/1_000_000:.2f} MH/s ({hashes:,} hashes in {elapsed:.1f}s)")
    return rate

def worker_bench(queue, test_string, duration):
    """Worker process for benchmark"""
    hashes = 0
    start = time.time()
    
    while time.time() - start < duration:
        for i in range(10000):
            msg = test_string + str(hashes + i)
            hashlib.sha256(msg.encode()).hexdigest()
        hashes += 10000
    
    queue.put(hashes)

def benchmark_multicore():
    """Benchmark all cores"""
    cores = multiprocessing.cpu_count()
    print(f"\nTesting {cores} cores performance...")
    
    test_string = "brvdgx5w3p00000001491c"
    duration = 5  # 5 second test
    
    # Create queue and processes
    queue = multiprocessing.Queue()
    processes = []
    
    start = time.time()
    for i in range(cores):
        p = multiprocessing.Process(target=worker_bench, args=(queue, test_string + str(i), duration))
        p.start()
        processes.append(p)
    
    # Wait for all to complete
    for p in processes:
        p.join()
    
    # Collect results
    total_hashes = 0
    while not queue.empty():
        total_hashes += queue.get()
    
    elapsed = time.time() - start
    rate = total_hashes / elapsed
    print(f"All {cores} cores: {rate/1_000_000:.2f} MH/s ({total_hashes:,} hashes in {elapsed:.1f}s)")
    print(f"Per core average: {rate/cores/1_000_000:.2f} MH/s")
    
    return rate

def main():
    print("=== CPU Mining Performance Test ===\n")
    
    # System info
    cores = multiprocessing.cpu_count()
    print(f"CPU cores detected: {cores}")
    
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    print(f"CPU model: {line.split(':')[1].strip()}")
                    break
    except:
        pass
    
    print()
    
    # Run benchmarks
    single_rate = benchmark_single_core()
    multi_rate = benchmark_multicore()
    
    # Analysis
    print(f"\nPerformance Analysis:")
    print(f"Scaling efficiency: {multi_rate / (single_rate * cores) * 100:.1f}%")
    print(f"Expected Bckn mining rate: ~{multi_rate/1_000_000:.0f} MH/s")
    print(f"Expected time to find block: ~{2**48 / 100000 / multi_rate:.0f} seconds")

if __name__ == "__main__":
    main()