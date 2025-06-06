#!/usr/bin/env python3
"""
Test if our mining calculations are correct
"""

import hashlib

# Test with known values
test_cases = [
    # (address, last_hash, nonce, expected_hash_start)
    ("brvdgx5w3p", "000000000abc", "0", None),
    ("brvdgx5w3p", "000000000abc", "1", None),
    ("brvdgx5w3p", "000000000abc", "12345", None),
]

print("Testing hash calculations...")
print("="*60)

for address, last_hash, nonce, expected in test_cases:
    message = address + last_hash + str(nonce)
    hash_result = hashlib.sha256(message.encode()).hexdigest()
    hash_value = int(hash_result[:12], 16)
    
    print(f"\nAddress: {address}")
    print(f"Last hash: {last_hash}")
    print(f"Nonce: {nonce}")
    print(f"Message: {message}")
    print(f"Full hash: {hash_result}")
    print(f"First 12: {hash_result[:12]}")
    print(f"Decimal value: {hash_value:,}")

# Now test finding a valid nonce with easier difficulty
print("\n" + "="*60)
print("Testing with easier difficulty (work = 10,000,000)...")
print("="*60)

address = "brvdgx5w3p"
last_hash = "000000000abc"
easy_work = 10_000_000
nonce = 0
found_count = 0

while found_count < 3:  # Find 3 valid nonces
    message = address + last_hash + str(nonce)
    hash_result = hashlib.sha256(message.encode()).hexdigest()
    hash_value = int(hash_result[:12], 16)
    
    if hash_value <= easy_work:
        found_count += 1
        print(f"\nFound #{found_count}!")
        print(f"Nonce: {nonce}")
        print(f"Hash: {hash_result}")
        print(f"Value: {hash_value:,} <= {easy_work:,}")
    
    if nonce % 1000000 == 0 and nonce > 0:
        print(f"Checked {nonce:,} nonces...")
    
    nonce += 1

print("\nTest complete!")