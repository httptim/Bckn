#!/usr/bin/env python3
"""
Bckn CPU-Optimized Miner
Maximum performance for CPU mining pods
"""

import hashlib
import requests
import time
import sys
import os
from multiprocessing import Pool, cpu_count, Value, Array
import signal
import random
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BCKN_NODE = "https://bckn.dev"

# Global stats (shared memory)
stats = None

def init_worker(shared_stats):
    """Initialize worker with shared stats"""
    global stats
    stats = shared_stats
    # Ignore SIGINT in workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def mine_range(args):
    """Mine a range of nonces - optimized for speed"""
    prefix, start, end, work = args
    prefix_bytes = prefix.encode('ascii')
    
    # Local counter for efficiency
    local_hashes = 0
    
    for nonce in range(start, end):
        # Inline the hash calculation for speed
        msg = prefix_bytes + str(nonce).encode('ascii')
        hash_bytes = hashlib.sha256(msg).digest()
        
        # Extract first 6 bytes as integer (faster than hex conversion)
        hash_int = int.from_bytes(hash_bytes[:6], 'big')
        
        local_hashes += 1
        
        # Check if valid (shift right to match 12 hex chars)
        if hash_int >> 8 <= work:
            # Double-check with proper hex conversion
            hash_hex = hash_bytes.hex()
            hash_int_verified = int(hash_hex[:12], 16)
            if hash_int_verified <= work:
                return (nonce, hash_int_verified, hash_hex)
        
        # Update global counter periodically
        if local_hashes % 100000 == 0:
            with stats.get_lock():
                stats[0] += local_hashes
            local_hashes = 0
    
    # Final update
    with stats.get_lock():
        stats[0] += local_hashes
    
    return None

def get_mining_info():
    """Get current work and last block info"""
    try:
        work_response = requests.get(f"{BCKN_NODE}/work", verify=False, timeout=5)
        work_data = work_response.json()
        work = work_data.get('work', 0)
        
        block_response = requests.get(f"{BCKN_NODE}/blocks/last", verify=False, timeout=5)
        block_data = block_response.json()
        
        if 'block' in block_data and 'hash' in block_data['block']:
            last_hash = block_data['block']['hash'][:12]
        else:
            last_hash = "000000000000"
        
        return work, last_hash
    except:
        return None, None

def submit_block(address, nonce):
    """Submit mining solution"""
    try:
        response = requests.post(f"{BCKN_NODE}/submit",
                               json={"address": address, "nonce": str(nonce)},
                               verify=False,
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"\n\n🎉 BLOCK FOUND! Nonce: {nonce}")
                print(f"   Block Hash: {data.get('block', {}).get('hash', 'Unknown')}")
                print(f"   Reward: {data.get('block', {}).get('value', 0)} BCN")
                return True
            else:
                error_msg = data.get('error', 'Unknown error')
                print(f"\n❌ Submission rejected: {error_msg}")
    except Exception as e:
        print(f"\n❌ Error submitting block: {e}")
    
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-cpu-optimized.py <private_key>")
        return
    
    private_key = sys.argv[1]
    
    # Login
    print("\n=== Bckn CPU-Optimized Miner ===")
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": private_key},
                           verify=False,
                           timeout=10)
    
    if response.status_code != 200:
        print("Invalid private key!")
        return
    
    address = response.json()['address']
    print(f"Mining with address: {address}")
    
    # Get initial work
    work, last_hash = get_mining_info()
    if not work:
        print("Failed to get work!")
        return
    
    cores = cpu_count()
    print(f"\nDetected {cores} CPU cores")
    print(f"Work: {work} | Last block: {last_hash}")
    print(f"Difficulty: 1 in {2**48 / work:,.0f} hashes")
    print(f"Expected time to find block: ~{2**48 / work / cores / 3_000_000:.0f} seconds\n")
    
    # Shared memory for stats
    stats = Array('i', [0, 0, 0])  # total_hashes, blocks_found, should_stop
    
    # Mining parameters
    chunk_size = 5_000_000  # 5M per chunk for better granularity
    prefix = address + last_hash
    
    # Start mining
    start_time = time.time()
    nonce_base = random.randint(0, 2**31)  # Random starting point
    
    print("Starting mining...\n")
    
    try:
        with Pool(cores, initializer=init_worker, initargs=(stats,)) as pool:
            while True:
                # Create work chunks
                tasks = []
                for i in range(cores * 2):  # Queue 2x cores for efficiency
                    start = nonce_base + i * chunk_size
                    end = start + chunk_size
                    tasks.append(pool.apply_async(mine_range, ((prefix, start, end, work),)))
                
                nonce_base += cores * 2 * chunk_size
                
                # Process results
                for task in tasks:
                    try:
                        result = task.get(timeout=30)
                        if result:
                            nonce, hash_int, hash_hex = result
                            print(f"\n💎 Found valid nonce! {nonce}")
                            print(f"   Hash: {hash_hex}")
                            print(f"   Value: {hash_int} <= {work}")
                            
                            if submit_block(address, nonce):
                                stats[1] += 1
                                
                                # Get new work
                                time.sleep(1)
                                new_work, new_hash = get_mining_info()
                                if new_work and (new_work != work or new_hash != last_hash):
                                    print(f"\nNew work: {new_work} | Last block: {new_hash}")
                                    work = new_work
                                    last_hash = new_hash
                                    prefix = address + last_hash
                                    stats[0] = 0  # Reset hash counter
                                    start_time = time.time()
                                    nonce_base = random.randint(0, 2**31)
                    except:
                        pass  # Task still running
                
                # Print stats
                elapsed = time.time() - start_time
                hashrate = stats[0] / elapsed if elapsed > 0 else 0
                print(f"\r[{cores} cores] Rate: {hashrate/1_000_000:.2f} MH/s | "
                      f"Total: {stats[0]/1_000_000_000:.3f}B | "
                      f"Blocks: {stats[1]} | "
                      f"Runtime: {int(elapsed)}s", end='', flush=True)
                
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    main()