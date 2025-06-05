#!/usr/bin/env python3
"""
Bckn Stable CPU Miner - Fixed overflow and performance issues
"""

import hashlib
import requests
import time
import sys
import random
import urllib3
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

# Disable warnings
urllib3.disable_warnings()

BCKN_NODE = "https://bckn.dev"

def hash_chunk(args):
    """Hash a chunk of nonces and return any valid ones"""
    prefix, start_nonce, chunk_size, work = args
    valid_nonces = []
    
    # Pre-encode prefix for efficiency
    prefix_bytes = prefix.encode('ascii')
    
    for i in range(chunk_size):
        nonce = start_nonce + i
        
        # Create message and hash it
        msg = prefix_bytes + str(nonce).encode('ascii')
        hash_result = hashlib.sha256(msg).hexdigest()
        hash_int = int(hash_result[:12], 16)
        
        if hash_int <= work:
            valid_nonces.append((nonce, hash_int, hash_result))
    
    return chunk_size, valid_nonces

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-stable.py <private_key>")
        return
    
    # Login
    print("=== Bckn Stable CPU Miner ===")
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": sys.argv[1]},
                           verify=False)
    if response.status_code != 200:
        print("Invalid private key!")
        return
        
    address = response.json()['address']
    print(f"Mining with address: {address}")
    
    # Get initial work
    work_resp = requests.get(f"{BCKN_NODE}/work", verify=False)
    work = work_resp.json()['work']
    
    block_resp = requests.get(f"{BCKN_NODE}/blocks/last", verify=False)
    block_data = block_resp.json()
    last_hash = block_data.get('block', {}).get('hash', '000000000000')[:12]
    
    cores = cpu_count()
    print(f"\nCPU cores: {cores}")
    print(f"Work: {work} | Last block: {last_hash}")
    print(f"Difficulty: 1 in {2**48 / work:,.0f} hashes")
    print("Starting...\n")
    
    # Mining setup
    prefix = address + last_hash
    chunk_size = 100_000  # 100k hashes per chunk
    nonce = random.randint(0, 2**31)
    
    total_hashes = 0
    blocks_found = 0
    start_time = time.time()
    last_update = start_time
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        # Keep a queue of futures
        futures = []
        
        while True:
            # Submit new work
            while len(futures) < cores * 2:
                future = executor.submit(hash_chunk, (prefix, nonce, chunk_size, work))
                futures.append(future)
                nonce += chunk_size
            
            # Check completed work
            done = []
            for future in as_completed(futures, timeout=0.1):
                done.append(future)
                hashes_done, valid_nonces = future.result()
                total_hashes += hashes_done
                
                # Check for valid nonces
                if valid_nonces:
                    for valid_nonce, hash_int, hash_full in valid_nonces:
                        print(f"\n💎 Found valid nonce! {valid_nonce}")
                        print(f"   Hash: {hash_full}")
                        print(f"   Value: {hash_int} <= {work}")
                        
                        # Submit
                        try:
                            resp = requests.post(f"{BCKN_NODE}/submit",
                                               json={"address": address, "nonce": str(valid_nonce)},
                                               verify=False)
                            if resp.json().get('success'):
                                print("   ✓ Block submitted successfully!")
                                blocks_found += 1
                                
                                # Get new work
                                time.sleep(1)
                                work = requests.get(f"{BCKN_NODE}/work", verify=False).json()['work']
                                last_hash = requests.get(f"{BCKN_NODE}/blocks/last", verify=False).json()['block']['hash'][:12]
                                prefix = address + last_hash
                                print(f"\nNew work: {work} | Last block: {last_hash}\n")
                                
                                # Reset stats
                                total_hashes = 0
                                start_time = time.time()
                            else:
                                print("   ✗ Submission failed")
                        except Exception as e:
                            print(f"   ✗ Error: {e}")
            
            # Remove completed futures
            for future in done:
                futures.remove(future)
            
            # Update stats
            current_time = time.time()
            if current_time - last_update >= 1.0:
                elapsed = current_time - start_time
                hashrate = total_hashes / elapsed if elapsed > 0 else 0
                
                print(f"\r[{cores} cores] {hashrate/1_000_000:.2f} MH/s | "
                      f"{total_hashes/1_000_000_000:.3f}B hashes | "
                      f"{blocks_found} blocks | "
                      f"{int(elapsed)}s", end='', flush=True)
                
                last_update = current_time

if __name__ == "__main__":
    main()