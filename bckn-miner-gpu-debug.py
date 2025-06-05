#!/usr/bin/env python3
"""
Bckn GPU Miner - Debug Version
"""

import hashlib
import requests
import time
import sys
import numpy as np

# Configuration
BCKN_NODE = "https://bckn.dev"

def verify_hash(address, last_hash, nonce, work):
    """Verify if a nonce produces a valid hash"""
    message = address + last_hash + str(nonce)
    hash_result = hashlib.sha256(message.encode()).hexdigest()
    hash_int = int(hash_result[:12], 16)
    print(f"\nVerifying nonce {nonce}:")
    print(f"  Message: {message}")
    print(f"  Hash: {hash_result}")
    print(f"  First 12 chars: {hash_result[:12]}")
    print(f"  As integer: {hash_int}")
    print(f"  Work target: {work}")
    print(f"  Valid: {hash_int <= work}")
    return hash_int <= work

def test_mining_range(address, last_hash, work, start, count):
    """Test a range of nonces to see if any are valid"""
    print(f"\nTesting {count} nonces starting from {start}...")
    found = []
    for nonce in range(start, start + count):
        message = address + last_hash + str(nonce)
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        hash_int = int(hash_result[:12], 16)
        if hash_int <= work:
            found.append((nonce, hash_int))
            print(f"  Found valid nonce! {nonce} -> {hash_int}")
    
    if not found:
        print(f"  No valid nonces found in range {start} to {start+count}")
    return found

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-gpu-debug.py <private_key>")
        return
    
    private_key = sys.argv[1]
    
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
    print(f"Expected probability: 1 in {2**48 / work:,.0f} hashes")
    
    # Test some ranges
    test_mining_range(address, last_hash, work, 0, 1000000)
    test_mining_range(address, last_hash, work, 1000000000, 1000000)
    test_mining_range(address, last_hash, work, 10000000000, 1000000)
    
    # Let's also verify a specific nonce if you want
    if len(sys.argv) > 2:
        test_nonce = int(sys.argv[2])
        verify_hash(address, last_hash, test_nonce, work)
    
    # Test the specific nonce that got stuck
    print("\nTesting the stuck nonce 1090519040:")
    verify_hash(address, last_hash, 1090519040, work)

if __name__ == "__main__":
    main()