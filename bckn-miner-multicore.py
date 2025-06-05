#!/usr/bin/env python3
"""
Bckn Multicore Miner - True parallel processing
Actually uses all CPU cores unlike threading
"""

import hashlib
import requests
import time
import sys
import os
from multiprocessing import Process, Queue, cpu_count
import signal

BCKN_NODE = "https://bckn.dev"

def hash_worker(worker_id, address, last_hash, work, start_nonce, result_queue, stats_queue):
    """Worker process that mines independently"""
    prefix = address + last_hash
    nonce = start_nonce + (worker_id * 1_000_000_000)  # Each worker gets different range
    hashes = 0
    
    while True:
        # Hash batch
        for _ in range(100000):
            message = prefix + str(nonce)
            hash_result = hashlib.sha256(message.encode()).hexdigest()
            hash_int = int(hash_result[:12], 16)
            
            if hash_int <= work:
                # Found valid nonce!
                result_queue.put((nonce, hash_int, hash_result))
                return
            
            nonce += 1
            hashes += 1
        
        # Report progress
        stats_queue.put(('hashes', hashes))
        hashes = 0

def stats_worker(stats_queue, num_workers):
    """Process that collects and displays stats"""
    total_hashes = 0
    start_time = time.time()
    
    while True:
        try:
            stat_type, value = stats_queue.get(timeout=1)
            if stat_type == 'hashes':
                total_hashes += value
                
            elapsed = time.time() - start_time
            hashrate = total_hashes / elapsed if elapsed > 0 else 0
            
            print(f"\r{hashrate/1_000_000:.2f} MH/s | {total_hashes/1_000_000:.0f}M | {int(elapsed)}s", 
                  end='', flush=True)
        except:
            pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-multicore.py <private_key>")
        return
    
    # Login
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": sys.argv[1]},
                           verify=False)
    if response.status_code != 200:
        print("Invalid private key!")
        return
    
    address = response.json()['address']
    
    # Get work info
    work_resp = requests.get(f"{BCKN_NODE}/work", verify=False)
    work = work_resp.json()['work']
    
    block_resp = requests.get(f"{BCKN_NODE}/blocks/last", verify=False)
    last_hash = block_resp.json().get('block', {}).get('hash', '000000000000')[:12]
    
    # Setup
    num_workers = cpu_count()
    print(f"Bckn Multicore Miner - {num_workers} processes")
    print(f"Address: {address}")
    print(f"Work: {work} | Last: {last_hash}")
    print("Starting...\n")
    
    # Create queues
    result_queue = Queue()
    stats_queue = Queue()
    
    # Start stats process
    stats_proc = Process(target=stats_worker, args=(stats_queue, num_workers))
    stats_proc.daemon = True
    stats_proc.start()
    
    # Start mining processes
    processes = []
    for i in range(num_workers):
        p = Process(target=hash_worker, 
                   args=(i, address, last_hash, work, 0, result_queue, stats_queue))
        p.start()
        processes.append(p)
    
    # Wait for result
    try:
        nonce, hash_int, hash_full = result_queue.get()
        
        print(f"\n\n💎 Found valid nonce! {nonce}")
        print(f"Hash: {hash_full}")
        print(f"Value: {hash_int} <= {work}")
        
        # Kill all workers
        for p in processes:
            p.terminate()
        
        # Submit
        print("\nSubmitting...")
        resp = requests.post(f"{BCKN_NODE}/submit",
                           json={"address": address, "nonce": str(nonce)},
                           verify=False)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('success'):
                print("✓ Block submitted successfully!")
                print(f"Block hash: {data.get('block', {}).get('hash', 'Unknown')}")
                print(f"Reward: {data.get('block', {}).get('value', 0)} BCN")
            else:
                print(f"✗ Submission failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"✗ HTTP error: {resp.status_code}")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()