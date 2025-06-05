#!/usr/bin/env python3
"""
Bckn Address Generator
Generates a Bckn address and private key pair
"""

import hashlib
import secrets

def sha256(data):
    return hashlib.sha256(data.encode()).hexdigest()

def double_sha256(data):
    return sha256(sha256(data))

def hex_to_base36(num):
    byte = 48 + (num // 7)
    if byte + 39 > 122:
        return chr(101)
    elif byte > 57:
        return chr(byte + 39)
    else:
        return chr(byte)

def make_v2_address(key, prefix='b'):
    chars = [''] * 9
    chain = prefix
    hash_val = double_sha256(key)
    
    for i in range(9):
        chars[i] = hash_val[:2]
        hash_val = double_sha256(hash_val)
    
    i = 0
    while i <= 8:
        index = int(hash_val[2*i:2*i+2], 16) % 9
        
        if chars[index] == '':
            hash_val = sha256(hash_val)
        else:
            chain += hex_to_base36(int(chars[index], 16))
            chars[index] = ''
            i += 1
    
    return chain

def generate_address():
    # Generate a secure random private key
    private_key = secrets.token_hex(32)
    
    # Generate the address from the private key
    address = make_v2_address(private_key)
    
    return private_key, address

if __name__ == "__main__":
    print("=== Bckn Address Generator ===")
    print()
    
    private_key, address = generate_address()
    
    print(f"Address:     {address}")
    print(f"Private Key: {private_key}")
    print()
    print("⚠️  IMPORTANT: Save your private key securely!")
    print("    Anyone with your private key can access your BCN!")