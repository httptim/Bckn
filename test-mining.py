#!/usr/bin/env python3
"""
Test mining to verify probability and find valid nonces
"""

import hashlib
import requests
import time
import sys

BCKN_NODE = "https://bckn.dev"

def test_sequential_mining(address, last_hash, work, start_nonce=0, count=10_000_000):
    """Test sequential nonces to find valid ones"""
    prefix = address + last_hash
    found = []
    
    print(f"Testing {count:,} nonces starting from {start_nonce}...")
    print(f"Prefix: {prefix}")
    print(f"Work target: {work}")
    print(f"Expected probability: 1 in {2**48 / work:,.0f}")
    
    start_time = time.time()
    
    for nonce in range(start_nonce, start_nonce + count):
        if nonce % 1_000_000 == 0:
            elapsed = time.time() - start_time
            rate = nonce / elapsed if elapsed > 0 else 0
            print(f"\rProgress: {nonce-start_nonce:,} / {count:,} ({rate/1_000_000:.2f} MH/s)", end='', flush=True)
        
        message = prefix + str(nonce)
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        hash_int = int(hash_result[:12], 16)
        
        if hash_int <= work:
            found.append((nonce, hash_int, hash_result))
            print(f"\n✓ Found valid nonce! {nonce} -> {hash_int} (hash: {hash_result})")
    
    elapsed = time.time() - start_time
    print(f"\n\nCompleted in {elapsed:.1f} seconds")
    print(f"Average: {count/elapsed/1_000_000:.2f} MH/s")
    print(f"Found {len(found)} valid nonces")
    
    return found

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test-mining.py <private_key> [start_nonce] [count]")
        return
    
    private_key = sys.argv[1]
    start_nonce = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
    
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
    print()
    
    # Test mining
    found = test_sequential_mining(address, last_hash, work, start_nonce, count)
    
    if found:
        print("\n=== Valid nonces found ===")
        for nonce, hash_int, hash_full in found:
            print(f"Nonce: {nonce}")
            print(f"  Hash value: {hash_int}")
            print(f"  Full hash: {hash_full}")
            print()
            
        # Try submitting the first one
        if input("Submit first nonce? (y/n): ").lower() == 'y':
            nonce = found[0][0]
            response = requests.post(f"{BCKN_NODE}/submit",
                                   json={"address": address, "nonce": str(nonce)})
            print(f"Submission response: {response.json()}")

if __name__ == "__main__":
    main()