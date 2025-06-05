#!/usr/bin/env python3
"""
Bckn Fast Miner - Optimized for cloud CPUs
Reduces overhead, maximizes performance
"""

import hashlib
import requests
import time
import sys
import os
import urllib3
from multiprocessing import Process, Queue, cpu_count, Event
import random

# Disable warnings
urllib3.disable_warnings()

BCKN_NODE = "https://bckn.dev"

def hash_worker(worker_id, num_workers, address, last_hash, work, start_nonce, found_event, result_queue):
    """Optimized worker - minimal overhead"""
    # Each worker gets a different range
    nonce = start_nonce + (worker_id * (2**32 // num_workers))
    prefix = (address + last_hash).encode('ascii')
    
    # Local variables for speed
    local_work = work
    batch_size = 1_000_000
    
    while not found_event.is_set():
        # Process large batch without reporting
        for _ in range(batch_size):
            # Inline everything for speed
            msg = prefix + str(nonce).encode('ascii')
            hash_bytes = hashlib.sha256(msg).digest()
            
            # Fast check on first 6 bytes
            hash_int_fast = int.from_bytes(hash_bytes[:6], 'big') >> 8
            
            if hash_int_fast <= local_work:
                # Verify with full check
                hash_hex = hash_bytes.hex()
                hash_int = int(hash_hex[:12], 16)
                if hash_int <= local_work:
                    result_queue.put((nonce, hash_int, hash_hex))
                    found_event.set()
                    return
            
            nonce += 1

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-fast.py <private_key>")
        return
    
    # Login
    print("Bckn Fast Miner")
    resp = requests.post(f"{BCKN_NODE}/login", json={"privatekey": sys.argv[1]}, verify=False)
    address = resp.json()['address']
    
    # Get work
    work = requests.get(f"{BCKN_NODE}/work", verify=False).json()['work']
    last_hash = requests.get(f"{BCKN_NODE}/blocks/last", verify=False).json().get('block', {}).get('hash', '000000000000')[:12]
    
    num_workers = cpu_count()
    print(f"CPUs: {num_workers} | Address: {address}")
    print(f"Work: {work} | Last: {last_hash}")
    
    blocks_found = 0
    session_start = time.time()
    
    while True:
        print(f"\nMining round {blocks_found + 1}...")
        
        # Setup for this round
        result_queue = Queue()
        found_event = Event()
        start_nonce = random.randint(0, 2**31)
        start_time = time.time()
        
        # Start workers
        processes = []
        for i in range(num_workers):
            p = Process(target=hash_worker, 
                       args=(i, num_workers, address, last_hash, work, start_nonce, found_event, result_queue))
            p.start()
            processes.append(p)
        
        # Simple progress indicator
        while not found_event.is_set():
            time.sleep(5)
            elapsed = time.time() - start_time
            estimated_hashes = num_workers * 3_000_000 * elapsed  # Assume 3 MH/s per core
            print(f"\r~{estimated_hashes/1_000_000:.0f}M hashes | {int(elapsed)}s", end='', flush=True)
        
        # Get result
        nonce, hash_int, hash_hex = result_queue.get()
        elapsed = time.time() - start_time
        
        # Clean up processes
        for p in processes:
            p.terminate()
            p.join()
        
        print(f"\n\n💎 Found! Nonce: {nonce}")
        print(f"Hash: {hash_hex} (value: {hash_int})")
        print(f"Time: {elapsed:.1f}s")
        
        # Submit
        resp = requests.post(f"{BCKN_NODE}/submit",
                           json={"address": address, "nonce": str(nonce)},
                           verify=False)
        
        if resp.json().get('success'):
            blocks_found += 1
            print(f"✓ Block #{blocks_found} submitted!")
            
            # Get new work
            time.sleep(1)
            work = requests.get(f"{BCKN_NODE}/work", verify=False).json()['work']
            last_hash = requests.get(f"{BCKN_NODE}/blocks/last", verify=False).json()['block']['hash'][:12]
            
            # Stats
            session_time = time.time() - session_start
            print(f"\nSession: {blocks_found} blocks in {int(session_time)}s")
            print(f"Average: {session_time/blocks_found:.1f}s per block")
        else:
            print("✗ Submission failed")

if __name__ == "__main__":
    main()