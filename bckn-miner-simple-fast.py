#!/usr/bin/env python3
"""
Simple fast Bckn miner - single threaded but optimized
Good baseline for testing
"""

import hashlib
import requests
import time
import sys
import random
import urllib3

# Disable warnings
urllib3.disable_warnings()

BCKN_NODE = "https://bckn.dev"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-simple-fast.py <private_key>")
        return
    
    # Login
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": sys.argv[1]},
                           verify=False)
    address = response.json()['address']
    
    # Get work
    work_response = requests.get(f"{BCKN_NODE}/work", verify=False)
    work = work_response.json()['work']
    
    block_response = requests.get(f"{BCKN_NODE}/blocks/last", verify=False)
    last_hash = block_response.json().get('block', {}).get('hash', '000000000000')[:12]
    
    print(f"Mining with address: {address}")
    print(f"Work: {work} | Last block: {last_hash}")
    print(f"Difficulty: 1 in {2**48 / work:,.0f}\n")
    
    # Mine
    prefix = (address + last_hash).encode('ascii')
    nonce = random.randint(0, 2**32)
    start_time = time.time()
    hashes = 0
    
    while True:
        # Fast hash check
        msg = prefix + str(nonce).encode('ascii')
        hash_bytes = hashlib.sha256(msg).digest()
        
        # Quick check using first 6 bytes
        if int.from_bytes(hash_bytes[:6], 'big') >> 8 <= work:
            # Verify with full check
            hash_hex = hash_bytes.hex()
            if int(hash_hex[:12], 16) <= work:
                print(f"\n💎 Found! Nonce: {nonce}, Hash: {hash_hex}")
                
                # Submit
                resp = requests.post(f"{BCKN_NODE}/submit",
                                   json={"address": address, "nonce": str(nonce)},
                                   verify=False)
                if resp.json().get('success'):
                    print("🎉 Block submitted successfully!")
                    
                    # Get new work
                    time.sleep(1)
                    work = requests.get(f"{BCKN_NODE}/work", verify=False).json()['work']
                    last_hash = requests.get(f"{BCKN_NODE}/blocks/last", verify=False).json()['block']['hash'][:12]
                    prefix = (address + last_hash).encode('ascii')
                    print(f"\nNew work: {work} | Last block: {last_hash}\n")
        
        nonce += 1
        hashes += 1
        
        # Stats every million hashes
        if hashes % 1_000_000 == 0:
            elapsed = time.time() - start_time
            rate = hashes / elapsed
            print(f"\r{rate/1_000_000:.2f} MH/s | {hashes/1_000_000:.0f}M hashes | {int(elapsed)}s", end='', flush=True)

if __name__ == "__main__":
    main()