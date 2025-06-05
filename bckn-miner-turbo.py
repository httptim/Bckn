#!/usr/bin/env python3
"""
Bckn Turbo Miner - Maximum performance for high-end CPUs
Optimized for 5.7 GHz CPUs with DDR5
"""

import hashlib
import requests
import time
import sys
import os
import urllib3
from multiprocessing import Process, Queue, Value, cpu_count
import ctypes
import random

# Disable warnings
urllib3.disable_warnings()

BCKN_NODE = "https://bckn.dev"

def ultra_fast_worker(worker_id, num_workers, prefix_bytes, work, start_nonce, counter, found_flag, result_data):
    """Ultra-optimized mining loop"""
    # Worker-specific setup
    nonce = start_nonce + (worker_id * 10_000_000_000)
    
    # Pre-allocate for speed
    sha256 = hashlib.sha256
    
    # Localize everything
    local_work = work
    local_counter = 0
    
    # Main mining loop - unrolled for speed
    while found_flag.value == 0:
        # Process 10 at a time (loop unrolling)
        for _ in range(100000):  # 1M total per outer loop
            # Unrolled 10x for CPU pipeline efficiency
            h1 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h1[:6], 'big') >> 8 <= local_work:
                hash_hex = h1.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h2 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h2[:6], 'big') >> 8 <= local_work:
                hash_hex = h2.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h3 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h3[:6], 'big') >> 8 <= local_work:
                hash_hex = h3.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h4 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h4[:6], 'big') >> 8 <= local_work:
                hash_hex = h4.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h5 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h5[:6], 'big') >> 8 <= local_work:
                hash_hex = h5.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h6 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h6[:6], 'big') >> 8 <= local_work:
                hash_hex = h6.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h7 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h7[:6], 'big') >> 8 <= local_work:
                hash_hex = h7.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h8 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h8[:6], 'big') >> 8 <= local_work:
                hash_hex = h8.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h9 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h9[:6], 'big') >> 8 <= local_work:
                hash_hex = h9.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            h10 = sha256(prefix_bytes + str(nonce).encode()).digest()
            if int.from_bytes(h10[:6], 'big') >> 8 <= local_work:
                hash_hex = h10.hex()
                if int(hash_hex[:12], 16) <= local_work:
                    with found_flag.get_lock():
                        if found_flag.value == 0:
                            found_flag.value = 1
                            result_data[0] = nonce
                            result_data[1] = int(hash_hex[:12], 16)
                    return
            nonce += 1
            
            local_counter += 10
        
        # Update shared counter occasionally
        if local_counter >= 10_000_000:
            with counter.get_lock():
                counter.value += local_counter
            local_counter = 0

def stats_printer(counter, start_time):
    """Print stats in separate process"""
    while True:
        time.sleep(1)
        elapsed = time.time() - start_time
        hashrate = counter.value / elapsed if elapsed > 0 else 0
        print(f"\r{hashrate/1_000_000:.2f} MH/s | {counter.value/1_000_000_000:.3f}B hashes | {int(elapsed)}s", 
              end='', flush=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-turbo.py <private_key>")
        return
    
    # Login
    print("=== Bckn Turbo Miner ===")
    resp = requests.post(f"{BCKN_NODE}/login", json={"privatekey": sys.argv[1]}, verify=False)
    address = resp.json()['address']
    
    # Get work
    work = requests.get(f"{BCKN_NODE}/work", verify=False).json()['work']
    last_hash = requests.get(f"{BCKN_NODE}/blocks/last", verify=False).json().get('block', {}).get('hash', '000000000000')[:12]
    
    num_workers = cpu_count()
    print(f"CPUs: {num_workers} | 5.7 GHz DDR5 optimized")
    print(f"Address: {address}")
    print(f"Work: {work} | Last: {last_hash}")
    print("Starting turbo mode...\n")
    
    # Shared memory
    counter = Value('q', 0)  # long long for hash counter
    found_flag = Value('i', 0)
    result_data = (ctypes.c_longlong * 2)()  # [nonce, hash_value]
    
    # Pre-encode prefix
    prefix_bytes = (address + last_hash).encode('ascii')
    
    # Start time
    start_time = time.time()
    counter.value = 0
    found_flag.value = 0
    
    # Start stats printer
    stats_proc = Process(target=stats_printer, args=(counter, start_time))
    stats_proc.daemon = True
    stats_proc.start()
    
    # Start workers
    processes = []
    start_nonce = random.randint(0, 2**31)
    
    for i in range(num_workers):
        p = Process(target=ultra_fast_worker, 
                   args=(i, num_workers, prefix_bytes, work, start_nonce, counter, found_flag, result_data))
        p.start()
        processes.append(p)
    
    # Wait for result
    while found_flag.value == 0:
        time.sleep(0.1)
    
    # Get result
    found_nonce = result_data[0]
    hash_value = result_data[1]
    
    # Kill workers
    for p in processes:
        p.terminate()
    stats_proc.terminate()
    
    # Verify and submit
    message = address + last_hash + str(found_nonce)
    hash_hex = hashlib.sha256(message.encode()).hexdigest()
    
    elapsed = time.time() - start_time
    final_hashrate = counter.value / elapsed
    
    print(f"\n\n💎 Found valid nonce! {found_nonce}")
    print(f"Hash: {hash_hex}")
    print(f"Value: {int(hash_hex[:12], 16)} <= {work}")
    print(f"Time: {elapsed:.1f}s at {final_hashrate/1_000_000:.2f} MH/s")
    
    # Submit
    resp = requests.post(f"{BCKN_NODE}/submit",
                       json={"address": address, "nonce": str(found_nonce)},
                       verify=False)
    
    if resp.json().get('success'):
        print("✓ Block submitted successfully!")
        print(f"Reward: 25 BCN")
    else:
        print(f"✗ Submission failed: {resp.json()}")

if __name__ == "__main__":
    main()