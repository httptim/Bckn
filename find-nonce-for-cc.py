#!/usr/bin/env python3
"""
Find a valid nonce for testing ComputerCraft submission
"""

import hashlib
import requests
import sys

BCKN_NODE = "https://bckn.dev"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 find-nonce-for-cc.py <private_key>")
        return
    
    # Login
    print("Logging in...")
    resp = requests.post(f"{BCKN_NODE}/login", json={"privatekey": sys.argv[1]}, verify=False)
    address = resp.json()['address']
    print(f"Address: {address}")
    
    # Get current work
    work_resp = requests.get(f"{BCKN_NODE}/work", verify=False).json()
    work = work_resp['work']
    last_block = requests.get(f"{BCKN_NODE}/blocks/last", verify=False).json()
    last_hash = last_block.get('block', {}).get('hash', '000000000000')[:12]
    block_height = last_block.get('block', {}).get('id', 'unknown')
    
    print(f"Work: {work}")
    print(f"Last hash: {last_hash}")
    print(f"Block height: {block_height}")
    print(f"Full work response: {work_resp}")
    print(f"\nSearching for valid nonce...\n")
    
    # Also print some close calls
    closest_value = float('inf')
    closest_nonce = 0
    zeros_count = 0
    
    # Start mining
    nonce = 0
    prefix = address + last_hash
    
    while True:
        message = prefix + str(nonce)
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        hash_value = int(hash_result[:12], 16)
        
        # Track closest
        if hash_value < closest_value:
            closest_value = hash_value
            closest_nonce = nonce
        
        # Count hashes starting with zeros
        if hash_result.startswith("00000"):
            zeros_count += 1
        
        if nonce % 1000000 == 0:
            print(f"Nonce {nonce}: Current work={work} | Closest: {closest_value} | Below 100k: {zeros_count}")
            
        # Print any that are close
        if hash_value < 100000:
            print(f">>> Close one! Nonce {nonce}: {hash_result[:12]} = {hash_value} (need <= {work})")
        
        if hash_value <= work:
            print("\n" + "="*50)
            print("FOUND VALID NONCE!")
            print("="*50)
            print(f"\nAddress: {address}")
            print(f"Last hash: {last_hash}")
            print(f"Nonce: {nonce}")
            print(f"Message: {message}")
            print(f"Hash: {hash_result}")
            print(f"Hash value: {hash_value} <= {work}")
            print(f"\nTo test in ComputerCraft, use these values:")
            print(f"Address: {address}")
            print(f"Nonce: {nonce}")
            print("\n" + "="*50)
            
            # Don't submit - let CC do it
            print("\nNOT submitting - use this nonce in ComputerCraft to test!")
            break
        
        nonce += 1

if __name__ == "__main__":
    main()