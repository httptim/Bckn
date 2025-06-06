#!/usr/bin/env python3
"""
Diagnose why mining isn't finding blocks
"""

import hashlib
import requests

BCKN_NODE = "https://bckn.dev"

# Get current state
print("Fetching current network state...")
work_resp = requests.get(f"{BCKN_NODE}/work", verify=False).json()
last_block_resp = requests.get(f"{BCKN_NODE}/blocks/last", verify=False).json()

work = work_resp['work']
last_block = last_block_resp['block']
last_hash = last_block['hash'][:12]

print(f"\nCurrent Network State:")
print(f"Work: {work:,}")
print(f"Block height: {last_block['height']}")
print(f"Last miner: {last_block['address']}")
print(f"Last hash (full): {last_block['hash']}")
print(f"Last hash (12 chars): {last_hash}")
print(f"Block difficulty: {last_block['difficulty']:,}")
print(f"Block value: {last_block['value']}")

# Test with your address
address = "brvdgx5w3p"
print(f"\nTesting with address: {address}")

# Calculate expected probability
max_12_chars = 16**12
probability = work / max_12_chars
expected_hashes = 1 / probability

print(f"\nProbability Analysis:")
print(f"Max value for 12 hex chars: {max_12_chars:,}")
print(f"Probability per hash: 1 in {int(1/probability):,}")
print(f"Expected hashes to find block: {int(expected_hashes):,}")

# Find some example valid nonces by trying with much higher work
print(f"\nSearching for nonces that would be valid at different work levels...")

nonce = 0
levels_found = {
    1_000_000: None,
    10_000_000: None,
    100_000_000: None,
    1_000_000_000: None
}

prefix = address + last_hash
print(f"Mining prefix: {prefix}")

while any(v is None for v in levels_found.values()) and nonce < 100_000_000:
    message = prefix + str(nonce)
    hash_result = hashlib.sha256(message.encode()).hexdigest()
    hash_value = int(hash_result[:12], 16)
    
    for level, found in levels_found.items():
        if found is None and hash_value <= level:
            levels_found[level] = (nonce, hash_value, hash_result)
            print(f"\nFound nonce valid at work={level:,}:")
            print(f"  Nonce: {nonce}")
            print(f"  Hash: {hash_result}")
            print(f"  Value: {hash_value:,}")
    
    if nonce % 1_000_000 == 0 and nonce > 0:
        print(f"Checked {nonce:,} nonces...")
    
    nonce += 1

print(f"\n\nSummary:")
print(f"Work level {work:,} requires a hash starting with approximately:")
work_hex = hex(work)[2:].zfill(12)
print(f"  Hex: {work_hex}")
print(f"  Pattern: Needs ~{12 - len(hex(work)[2:])} leading zeros")