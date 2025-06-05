#!/usr/bin/env python3
"""
Krist CPU Miner - Simple & Fast
For macOS/Apple Silicon
"""

import hashlib
import time
import sys
import os
from multiprocessing import Process, Queue, cpu_count
from datetime import datetime

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

KRIST_NODE = "https://bckn.dev"

def sha256(data):
    return hashlib.sha256(data.encode()).hexdigest()

def mine_worker(worker_id, address, last_hash, work, result_queue):
    """Pure mining function - no imports needed"""
    nonce = worker_id
    step = cpu_count()
    count = 0
    
    while True:
        # Hash
        h = sha256(address + last_hash + str(nonce))
        val = int(h[:12], 16)
        
        # Found it!
        if val <= work:
            result_queue.put((nonce, h))
            return
            
        nonce += step
        count += 1
        
        # Progress update
        if count % 1000000 == 0:
            result_queue.put(('progress', count))
            count = 0

def get_api(endpoint):
    """Simple API getter"""
    import urllib.request
    import ssl
    import json
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(KRIST_NODE + endpoint, context=ctx) as response:
        return json.loads(response.read())

def post_api(endpoint, data):
    """Simple API poster"""
    import urllib.request
    import ssl
    import json
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        KRIST_NODE + endpoint,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req, context=ctx) as response:
        return json.loads(response.read())

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 krist-miner-simple.py <private_key>")
        return
        
    # Login
    private_key = sys.argv[1]
    login = post_api("/login", {"privatekey": private_key})
    address = login['address']
    
    print(f"Krist Miner - {cpu_count()} threads")
    print(f"Address: {address}")
    
    # Get work info
    work_data = get_api("/work")
    work = work_data['work']
    
    # Get last block
    try:
        block_data = get_api("/blocks/last")
        last_hash = block_data['block']['hash'][:12]
    except:
        last_hash = "000000000000"  # Genesis
        print("Mining genesis block!")
    
    print(f"Work: {work} | Last: {last_hash}")
    print(f"Starting...\n")
    
    # Setup
    result_queue = Queue()
    processes = []
    start_time = time.time()
    total_hashes = 0
    
    # Start miners
    for i in range(cpu_count()):
        p = Process(target=mine_worker, args=(i, address, last_hash, work, result_queue))
        p.start()
        processes.append(p)
    
    # Monitor
    try:
        while True:
            if not result_queue.empty():
                result = result_queue.get()
                
                if result[0] == 'progress':
                    total_hashes += result[1]
                else:
                    # Found!
                    nonce, hash_val = result
                    
                    # Kill workers
                    for p in processes:
                        p.terminate()
                    
                    print(f"\n💎 FOUND! Nonce: {nonce}")
                    print(f"Hash: {hash_val}")
                    
                    # Submit
                    try:
                        submit = post_api("/submit", {"address": address, "nonce": str(nonce)})
                        if submit.get('success'):
                            print("✅ Block mined successfully!")
                            print(f"Reward: {submit['block']['value']} KST")
                        else:
                            print("❌ Submission failed:", submit)
                    except Exception as e:
                        print("❌ Error submitting:", e)
                    
                    return
            
            # Stats
            elapsed = time.time() - start_time
            if elapsed > 0 and total_hashes > 0:
                rate = total_hashes / elapsed
                # Calculate ETA (2.8 billion hashes expected)
                remaining = 2814749767 - total_hashes
                eta_seconds = remaining / rate if rate > 0 else 999999
                if eta_seconds < 60:
                    eta = f"{int(eta_seconds)}s"
                elif eta_seconds < 3600:
                    eta = f"{int(eta_seconds/60)}m"
                else:
                    eta = f"{eta_seconds/3600:.1f}h"
                print(f"\r{rate/1e6:.2f} MH/s | {total_hashes/1e6:.0f}M | {int(elapsed)}s | ETA: {eta}", 
                      end='', flush=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        for p in processes:
            p.terminate()

if __name__ == '__main__':
    # Required for macOS multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()