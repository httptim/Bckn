#!/usr/bin/env python3
"""
Fast parallel search for valid nonces
"""

import hashlib
import requests
import time
import sys
from multiprocessing import Pool, cpu_count
import random

BCKN_NODE = "https://bckn.dev"

def check_nonce_range(args):
    """Check a range of nonces for validity"""
    prefix, start, end, work = args
    found = []
    
    for nonce in range(start, end):
        message = prefix + str(nonce)
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        hash_int = int(hash_result[:12], 16)
        
        if hash_int <= work:
            found.append((nonce, hash_int, hash_result))
    
    return found

def find_valid_nonces(address, last_hash, work, num_samples=100):
    """
    Find valid nonces by testing random samples across the nonce space
    """
    prefix = address + last_hash
    cores = cpu_count()
    
    print(f"Searching for valid nonces using {cores} CPU cores...")
    print(f"Testing {num_samples} random ranges of 10M each\n")
    
    all_found = []
    
    with Pool(cores) as pool:
        for i in range(num_samples):
            # Test random starting points across the nonce space
            base = random.randint(0, 2**32)
            chunk_size = 10_000_000 // cores
            
            # Create work for each core
            tasks = []
            for j in range(cores):
                start = base + j * chunk_size
                end = start + chunk_size
                tasks.append((prefix, start, end, work))
            
            # Process in parallel
            results = pool.map(check_nonce_range, tasks)
            
            # Collect results
            for result in results:
                all_found.extend(result)
            
            print(f"\rProgress: {i+1}/{num_samples} samples, found {len(all_found)} valid nonces", end='', flush=True)
            
            # If we found some, show them
            if all_found and len(all_found) <= 5:
                print(f"\n\nFound valid nonces:")
                for nonce, hash_int, _ in all_found:
                    print(f"  Nonce: {nonce} -> hash: {hash_int}")
                print()
    
    return all_found

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 find-valid-nonce.py <private_key> [num_samples]")
        return
    
    private_key = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Login
    response = requests.post(f"{BCKN_NODE}/login", json={"privatekey": private_key})
    if response.status_code != 200:
        print("Invalid private key!")
        return
    
    address = response.json()['address']
    print(f"Address: {address}")
    
    # Get work
    work_response = requests.get(f"{BCKN_NODE}/work")
    work = work_response.json()['work']
    
    block_response = requests.get(f"{BCKN_NODE}/blocks/last")
    block_data = block_response.json()
    if 'block' in block_data:
        last_hash = block_data['block']['hash'][:12]
    else:
        last_hash = "000000000000"
    
    print(f"Work: {work}")
    print(f"Last hash: {last_hash}")
    print(f"Expected probability: 1 in {2**48 / work:,.0f}")
    print()
    
    # Find valid nonces
    start_time = time.time()
    found = find_valid_nonces(address, last_hash, work, num_samples)
    elapsed = time.time() - start_time
    
    print(f"\n\nCompleted in {elapsed:.1f} seconds")
    print(f"Tested {num_samples * 10_000_000:,} nonces total")
    print(f"Found {len(found)} valid nonces")
    
    if found:
        print("\n=== First 10 valid nonces ===")
        for i, (nonce, hash_int, hash_full) in enumerate(found[:10]):
            print(f"{i+1}. Nonce: {nonce}")
            print(f"   Hash: {hash_full}")
            print(f"   Value: {hash_int} (target: {work})")
        
        # Test submit
        print(f"\nTesting submission with nonce {found[0][0]}...")
        response = requests.post(f"{BCKN_NODE}/submit",
                               json={"address": address, "nonce": str(found[0][0])})
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    main()