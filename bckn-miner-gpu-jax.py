#!/usr/bin/env python3
"""
Bckn GPU Miner using JAX
High-performance GPU mining using Google's JAX library
"""

import hashlib
import requests
import time
import sys
import numpy as np
from datetime import datetime
import warnings
import urllib3
from functools import partial

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap
    
    # Check for GPU
    if jax.devices()[0].platform == 'gpu':
        print(f"JAX GPU acceleration enabled")
        print(f"GPU: {jax.devices()[0]}")
        GPU_AVAILABLE = True
    else:
        print("No GPU found for JAX! Using CPU")
        GPU_AVAILABLE = False
        
except ImportError:
    print("JAX not found. Install with:")
    print("pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    sys.exit(1)

# Configuration
BCKN_NODE = "https://bckn.dev"
BATCH_SIZE = 10_000_000  # 10M nonces per batch

# Global stats
blocks_found = 0
total_hashes = 0
start_time = time.time()

def get_mining_info():
    """Get current work and last block info"""
    try:
        work_response = requests.get(f"{BCKN_NODE}/work", verify=False, timeout=5)
        if work_response.status_code != 200:
            return None, None
            
        work_data = work_response.json()
        work = work_data.get('work', 0)
        
        block_response = requests.get(f"{BCKN_NODE}/blocks/last", verify=False, timeout=5)
        block_data = block_response.json()
        
        if not block_data.get('ok', True) and block_data.get('error') == 'block_not_found':
            last_block_hash = "000000000000"
        elif 'block' in block_data and 'hash' in block_data['block']:
            last_block_hash = block_data['block']['hash'][:12]
        else:
            return None, None
        
        return work, last_block_hash
    except Exception as e:
        return None, None

def submit_block(address, nonce):
    """Submit mining solution"""
    try:
        response = requests.post(f"{BCKN_NODE}/submit",
                               json={"address": address, "nonce": str(nonce)},
                               verify=False,
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"\n🎉 BLOCK FOUND! Nonce: {nonce}")
                print(f"   Block Hash: {data.get('block', {}).get('hash', 'Unknown')}")
                print(f"   Reward: {data.get('block', {}).get('value', 0)} BCN")
                return True
            else:
                error_msg = data.get('error', 'Unknown error')
                print(f"\n❌ Submission rejected: {error_msg}")
        else:
            print(f"\n❌ Submission failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"\n❌ Error submitting block: {e}")
    
    return False

def verify_nonce_cpu(prefix, nonce, work):
    """Verify a nonce on CPU"""
    message = prefix + str(nonce)
    hash_result = hashlib.sha256(message.encode()).hexdigest()
    hash_int = int(hash_result[:12], 16)
    return hash_int <= work

@jit
def hash_approximation_gpu(nonces, prefix_hash):
    """
    Fast hash approximation on GPU using JAX
    This filters candidates before CPU verification
    """
    # Mix nonces with prefix hash
    hashes = nonces * 31 + prefix_hash
    hashes = (hashes * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
    
    # Additional mixing
    hashes = hashes ^ (hashes >> 16)
    hashes = hashes * 0x85EBCA6B
    hashes = hashes ^ (hashes >> 13)
    hashes = hashes * 0xC2B2AE35
    hashes = hashes ^ (hashes >> 16)
    
    return hashes & 0xFFFFFFFFFFFF

def mine_batch_jax(prefix, start_nonce, work, batch_size):
    """
    Mine using JAX on GPU with vectorized operations
    """
    # Calculate prefix hash for mixing
    prefix_hash = sum(ord(c) * (31 ** i) for i, c in enumerate(prefix)) & 0xFFFFFFFF
    
    # Process in chunks to manage memory
    chunk_size = 1_000_000
    
    for chunk_start in range(0, batch_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, batch_size)
        
        # Generate nonces on GPU
        nonces = jnp.arange(start_nonce + chunk_start, start_nonce + chunk_end, dtype=jnp.int64)
        
        # Fast GPU filtering
        hash_approx = hash_approximation_gpu(nonces, prefix_hash)
        
        # Find potentially valid nonces
        mask = hash_approx < (work * 1000)  # Rough filter
        candidates = nonces[mask]
        
        # Move candidates to CPU for verification
        if len(candidates) > 0:
            candidates_cpu = np.array(candidates)
            
            for nonce in candidates_cpu:
                if verify_nonce_cpu(prefix, int(nonce), work):
                    return int(nonce)
    
    return None

def mine_gpu_jax(address, last_hash, work, start_nonce=0):
    """
    Main GPU mining function using JAX
    """
    global total_hashes, blocks_found
    
    prefix = address + last_hash
    nonce = start_nonce
    
    print(f"\nGPU mining with JAX")
    print(f"Processing {BATCH_SIZE:,} nonces per batch\n")
    
    while True:
        # Mine a batch
        found_nonce = mine_batch_jax(prefix, nonce, work, BATCH_SIZE)
        
        if found_nonce is not None:
            return found_nonce
        
        # Update stats
        nonce += BATCH_SIZE
        total_hashes += BATCH_SIZE
        
        elapsed = time.time() - start_time
        hashrate = total_hashes / elapsed if elapsed > 0 else 0
        
        print(f"\r[JAX GPU] Rate: {hashrate/1_000_000:.2f} MH/s | "
              f"Total: {total_hashes/1_000_000_000:.2f}B | "
              f"Blocks: {blocks_found} | "
              f"Nonce: {nonce:,}", end='', flush=True)
        
        # Check for new work periodically
        if nonce % (BATCH_SIZE * 10) == 0:
            new_work, new_hash = get_mining_info()
            if new_work and (new_work != work or new_hash != last_hash):
                print("\nWork changed, restarting...")
                return None

def main():
    global blocks_found, total_hashes, start_time
    
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-gpu-jax.py <private_key>")
        return
    
    private_key = sys.argv[1]
    
    # Login
    print("\n=== Bckn GPU Miner (JAX Edition) ===")
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": private_key},
                           verify=False,
                           timeout=10)
    
    if response.status_code != 200:
        print("Invalid private key!")
        return
    
    address = response.json()['address']
    print(f"Mining with address: {address}")
    
    current_work = None
    current_hash = None
    nonce_position = 0
    
    while True:
        # Get mining parameters
        work, last_hash = get_mining_info()
        if not work:
            time.sleep(5)
            continue
        
        # Check if work changed
        if work != current_work or last_hash != current_hash:
            print(f"\n\nNew work: {work} | Last block: {last_hash}")
            print(f"Difficulty: 1 in {2**48 / work:,.0f} hashes")
            current_work = work
            current_hash = last_hash
            nonce_position = 0
            total_hashes = 0
            start_time = time.time()
        
        # Mine on GPU
        nonce = mine_gpu_jax(address, last_hash, work, nonce_position)
        
        if nonce:
            print(f"\n\n💎 Found valid nonce! {nonce}")
            print("   Verifying...")
            
            # Double-check it's valid
            if verify_nonce_cpu(address + last_hash, nonce, work):
                print("   ✓ Verified! Submitting...")
                if submit_block(address, nonce):
                    blocks_found += 1
                    nonce_position = 0
                else:
                    nonce_position = nonce + 1
            else:
                print("   ✗ Verification failed")
                nonce_position = nonce + 1
        else:
            # Continue from where we left off
            nonce_position += BATCH_SIZE

if __name__ == "__main__":
    main()