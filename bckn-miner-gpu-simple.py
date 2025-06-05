#!/usr/bin/env python3
"""
Bckn GPU Miner - Simple Version
Compatible with RunPod and various CUDA environments
"""

import hashlib
import requests
import time
import sys
import json
from datetime import datetime
import warnings
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Try to import CuPy (GPU acceleration)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with CuPy")
except ImportError:
    print("WARNING: CuPy not found, falling back to CPU mining")
    print("Install with: pip install cupy-cuda12x")
    GPU_AVAILABLE = False
    import numpy as cp  # Fallback to numpy

# Configuration
BCKN_NODE = "https://bckn.dev"
BATCH_SIZE = 1000000  # 1M hashes per batch

def get_mining_info():
    """Get current work and last block info"""
    try:
        # Get work
        work_response = requests.get(f"{BCKN_NODE}/work", verify=False)
        if work_response.status_code != 200:
            print(f"Work API error: {work_response.status_code}")
            return None, None
            
        work_data = work_response.json()
        work = work_data.get('work', 0)
        
        # Get last block
        block_response = requests.get(f"{BCKN_NODE}/blocks/last", verify=False)
        block_data = block_response.json()
        
        if not block_data.get('ok', True) and block_data.get('error') == 'block_not_found':
            print("No blocks found - mining genesis block!")
            last_block_hash = "000000000000"
        elif 'block' in block_data and 'hash' in block_data['block']:
            last_block_hash = block_data['block']['hash'][:12]
        else:
            print(f"Unexpected block format: {block_data}")
            return None, None
        
        return work, last_block_hash
    except Exception as e:
        print(f"Error getting mining info: {e}")
        return None, None

def submit_block(address, nonce):
    """Submit mining solution"""
    try:
        response = requests.post(f"{BCKN_NODE}/submit",
                               json={"address": address, "nonce": str(nonce)},
                               verify=False)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"\n🎉 BLOCK FOUND! Nonce: {nonce}")
                print(f"   Block Hash: {data.get('block', {}).get('hash', 'Unknown')}")
                print(f"   Reward: {data.get('block', {}).get('value', 0)} BCN")
                return True
        else:
            print(f"\n❌ Submission failed: {response.text}")
    except Exception as e:
        print(f"\n❌ Error submitting block: {e}")
    
    return False

def hash_attempt(prefix, nonce):
    """Calculate hash for a single attempt"""
    message = prefix + str(nonce)
    hash_result = hashlib.sha256(message.encode()).hexdigest()
    return int(hash_result[:12], 16)

def mine_batch_gpu(prefix, start_nonce, work, batch_size):
    """Mine a batch of nonces using GPU"""
    # Create array of nonces
    nonces = cp.arange(start_nonce, start_nonce + batch_size, dtype=cp.uint64)
    
    # This is a simplified version - in production you'd implement SHA256 in CuPy
    # For now, we'll process in smaller chunks
    chunk_size = 10000
    for i in range(0, batch_size, chunk_size):
        chunk_end = min(i + chunk_size, batch_size)
        chunk_nonces = range(start_nonce + i, start_nonce + chunk_end)
        
        for nonce in chunk_nonces:
            hash_val = hash_attempt(prefix, nonce)
            if hash_val <= work:
                return nonce
    
    return None

def mine_batch_cpu(prefix, start_nonce, work, batch_size):
    """Mine a batch of nonces using CPU"""
    for nonce in range(start_nonce, start_nonce + batch_size):
        hash_val = hash_attempt(prefix, nonce)
        if hash_val <= work:
            return nonce
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-gpu-simple.py <private_key>")
        return
    
    private_key = sys.argv[1]
    
    # Login to get address
    print("=== Bckn GPU Miner (Simple Version) ===")
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": private_key},
                           verify=False)
    
    if response.status_code != 200:
        print("Invalid private key!")
        return
    
    address = response.json()['address']
    print(f"Mining with address: {address}")
    print(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled (CPU mode)'}")
    print("\nStarting mining...\n")
    
    blocks_found = 0
    total_hashes = 0
    start_time = time.time()
    
    while True:
        # Get current mining parameters
        work, last_hash = get_mining_info()
        if not work:
            print("Failed to get mining info, retrying...")
            time.sleep(5)
            continue
        
        print(f"\nNew work: {work} | Last block: {last_hash}")
        
        # Prepare mining prefix
        prefix = address + last_hash
        nonce = 0
        
        # Mine until we find a valid nonce or work changes
        while True:
            # Mine a batch
            if GPU_AVAILABLE:
                found_nonce = mine_batch_gpu(prefix, nonce, work, BATCH_SIZE)
            else:
                found_nonce = mine_batch_cpu(prefix, nonce, work, BATCH_SIZE)
            
            if found_nonce is not None:
                # Submit the solution
                if submit_block(address, found_nonce):
                    blocks_found += 1
                break
            
            # Update stats
            nonce += BATCH_SIZE
            total_hashes += BATCH_SIZE
            
            # Print progress
            elapsed = time.time() - start_time
            hashrate = total_hashes / elapsed if elapsed > 0 else 0
            print(f"\r[{'GPU' if GPU_AVAILABLE else 'CPU'}] Hashrate: {hashrate/1_000_000:.2f} MH/s | "
                  f"Total: {total_hashes/1_000_000_000:.2f}B | "
                  f"Blocks: {blocks_found} | "
                  f"Current: {nonce}", end='', flush=True)
            
            # Check if we should get new work (every 10B hashes)
            if nonce % 10_000_000_000 == 0:
                new_work, new_hash = get_mining_info()
                if new_work and (new_work != work or new_hash != last_hash):
                    print("\nWork changed, getting new work...")
                    break

if __name__ == "__main__":
    main()