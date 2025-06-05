#!/usr/bin/env python3
"""
Bckn Ultimate Miner - Guaranteed to find blocks
Uses all available CPU cores for maximum performance
"""

import hashlib
import requests
import time
import sys
import os
from multiprocessing import Pool, cpu_count, Value, Manager
import signal
import random

BCKN_NODE = "https://bckn.dev"

# Shared variables for multiprocessing
total_hashes = None
blocks_found = None
should_stop = None

def init_worker(counter, blocks, stop_flag):
    """Initialize worker process with shared variables"""
    global total_hashes, blocks_found, should_stop
    total_hashes = counter
    blocks_found = blocks
    should_stop = stop_flag

def mine_chunk(args):
    """Mine a chunk of nonces"""
    prefix, start_nonce, chunk_size, work = args
    
    for nonce in range(start_nonce, start_nonce + chunk_size):
        if should_stop.value:
            return None
            
        message = prefix + str(nonce)
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        hash_int = int(hash_result[:12], 16)
        
        if hash_int <= work:
            return (nonce, hash_int, hash_result)
        
        # Update counter
        if nonce % 10000 == 0:
            with total_hashes.get_lock():
                total_hashes.value += 10000
    
    return None

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
                print(f"\n🎉 BLOCK FOUND! Nonce: {nonce}")
                print(f"   Block Hash: {data.get('block', {}).get('hash', 'Unknown')}")
                print(f"   Reward: {data.get('block', {}).get('value', 0)} BCN")
                return True
            else:
                error_msg = data.get('error', 'Unknown error')
                print(f"\n❌ Submission rejected: {error_msg}")
        else:
            print(f"\n❌ Submission failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"\n❌ Error submitting block: {e}")
    
    return False

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

def mine_parallel(address, last_hash, work, start_nonce=0):
    """Mine using all CPU cores in parallel"""
    prefix = address + last_hash
    cores = cpu_count()
    chunk_size = 1_000_000  # 1M per chunk
    
    print(f"\nMining with {cores} CPU cores")
    print(f"Expected to find block every {2**48 / work / cores / 3_000_000:,.1f} seconds\n")
    
    # Reset stop flag
    should_stop.value = 0
    
    with Pool(cores, initializer=init_worker, initargs=(total_hashes, blocks_found, should_stop)) as pool:
        nonce = start_nonce
        tasks = []
        
        while not should_stop.value:
            # Create tasks for each core
            new_tasks = []
            for i in range(cores):
                task_args = (prefix, nonce + i * chunk_size, chunk_size, work)
                new_tasks.append(pool.apply_async(mine_chunk, (task_args,)))
            
            tasks.extend(new_tasks)
            nonce += cores * chunk_size
            
            # Check completed tasks
            completed = []
            for task in tasks:
                if task.ready():
                    result = task.get()
                    if result:
                        should_stop.value = 1
                        pool.terminate()
                        return result
                    completed.append(task)
            
            # Remove completed tasks
            for task in completed:
                tasks.remove(task)
            
            # Limit pending tasks
            while len(tasks) > cores * 2:
                time.sleep(0.01)

def print_stats(start_time):
    """Print mining statistics"""
    while True:
        time.sleep(1)
        elapsed = time.time() - start_time
        hashrate = total_hashes.value / elapsed if elapsed > 0 else 0
        
        print(f"\r[CPU] Hashrate: {hashrate/1_000_000:.2f} MH/s | "
              f"Total: {total_hashes.value/1_000_000_000:.3f}B | "
              f"Blocks: {blocks_found.value} | "
              f"Runtime: {int(elapsed)}s", end='', flush=True)

def main():
    global total_hashes, blocks_found, should_stop
    
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-ultimate.py <private_key>")
        return
    
    private_key = sys.argv[1]
    
    # Login
    print("\n=== Bckn Ultimate Miner ===")
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": private_key},
                           verify=False,
                           timeout=10)
    
    if response.status_code != 200:
        print("Invalid private key!")
        return
    
    address = response.json()['address']
    print(f"Mining with address: {address}")
    
    # Initialize shared variables
    manager = Manager()
    total_hashes = Value('i', 0)
    blocks_found = Value('i', 0)
    should_stop = Value('i', 0)
    
    # Get initial work
    work, last_hash = get_mining_info()
    if not work:
        print("Failed to get work!")
        return
        
    print(f"\nWork: {work} | Last block: {last_hash}")
    print(f"Difficulty: 1 in {2**48 / work:,.0f} hashes")
    
    # Start stats thread
    import threading
    start_time = time.time()
    stats_thread = threading.Thread(target=print_stats, args=(start_time,), daemon=True)
    stats_thread.start()
    
    # Main mining loop
    nonce_position = random.randint(0, 2**32)  # Start at random position
    
    while True:
        # Mine
        result = mine_parallel(address, last_hash, work, nonce_position)
        
        if result:
            nonce, hash_int, hash_full = result
            print(f"\n\n💎 Found valid nonce! {nonce}")
            print(f"   Hash: {hash_full}")
            print(f"   Value: {hash_int} (target: {work})")
            
            # Submit
            if submit_block(address, nonce):
                blocks_found.value += 1
                
                # Get new work
                time.sleep(2)
                new_work, new_hash = get_mining_info()
                if new_work and (new_work != work or new_hash != last_hash):
                    print(f"\nNew work: {new_work} | Last block: {new_hash}")
                    work = new_work
                    last_hash = new_hash
                    total_hashes.value = 0
                    start_time = time.time()
                    nonce_position = random.randint(0, 2**32)
                else:
                    nonce_position = nonce + 1_000_000
            else:
                # Continue from next position
                nonce_position = nonce + 1_000_000
        else:
            # No result, continue from new random position
            nonce_position = random.randint(0, 2**32)

if __name__ == "__main__":
    main()