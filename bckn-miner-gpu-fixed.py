#!/usr/bin/env python3
"""
Bckn GPU Miner for NVIDIA H100 - Fixed Version
Optimized for Digital Ocean GPU Droplet with 8x H100
"""

import hashlib
import requests
import json
import time
import numpy as np
from datetime import datetime
import signal
import sys
import os
import warnings
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Check for GPU libraries
try:
    import cupy as cp
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    GPU_AVAILABLE = True
except ImportError:
    print("GPU libraries not found. Install with:")
    print("pip install cupy-cuda12x pycuda")
    GPU_AVAILABLE = False
    sys.exit(1)

# Configuration
BCKN_NODE = "https://bckn.dev"
PRIVATE_KEY = None
ADDRESS = None
GPU_BATCH_SIZE = 1024 * 1024 * 16  # 16M hashes per batch per GPU
NUM_GPUS = cuda.Device.count()

# Global stats
blocks_found = 0
total_hashes = 0
start_time = time.time()
last_work_check = 0

# CUDA kernel for SHA256 mining
cuda_code = """
#include <stdint.h>

__device__ __constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_transform(uint32_t* state, const unsigned char* data) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h, t1, t2;
    
    // Load data into w[0..15]
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i*4] << 24) | (data[i*4 + 1] << 16) | 
               (data[i*4 + 2] << 8) | data[i*4 + 3];
    }
    
    // Extend w[16..63]
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    
    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__global__ void mine_kernel(const char* prefix, int prefix_len, 
                           uint64_t start_nonce, uint64_t work_target,
                           uint64_t* result_nonce, int* found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;
    
    // Prepare message
    char message[256];
    int msg_len = prefix_len;
    
    // Copy prefix
    for (int i = 0; i < prefix_len; i++) {
        message[i] = prefix[i];
    }
    
    // Add nonce
    char nonce_str[25];
    int nonce_len = 0;
    uint64_t n = nonce;
    
    // Convert nonce to string
    if (n == 0) {
        nonce_str[0] = '0';
        nonce_len = 1;
    } else {
        while (n > 0 && nonce_len < 24) {
            nonce_str[nonce_len++] = '0' + (n % 10);
            n /= 10;
        }
        // Reverse the string
        for (int i = 0; i < nonce_len / 2; i++) {
            char temp = nonce_str[i];
            nonce_str[i] = nonce_str[nonce_len - 1 - i];
            nonce_str[nonce_len - 1 - i] = temp;
        }
    }
    
    // Append nonce to message
    for (int i = 0; i < nonce_len; i++) {
        message[msg_len + i] = nonce_str[i];
    }
    msg_len += nonce_len;
    
    // SHA256 padding
    unsigned char padded[128];
    for (int i = 0; i < msg_len; i++) {
        padded[i] = message[i];
    }
    padded[msg_len] = 0x80;
    
    int padding_len = (msg_len < 56) ? (56 - msg_len) : (120 - msg_len);
    for (int i = 1; i < padding_len; i++) {
        padded[msg_len + i] = 0;
    }
    
    // Append length
    uint64_t bit_len = msg_len * 8;
    for (int i = 0; i < 8; i++) {
        padded[msg_len + padding_len + i] = (bit_len >> (56 - i * 8)) & 0xff;
    }
    
    // Initialize SHA256 state
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Process blocks
    int total_len = msg_len + padding_len + 8;
    for (int i = 0; i < total_len; i += 64) {
        sha256_transform(state, padded + i);
    }
    
    // Convert hash to hex string to extract first 12 chars
    char hex_hash[13];
    for (int i = 0; i < 3; i++) {
        uint32_t val = state[i];
        for (int j = 0; j < 4; j++) {
            if (i * 4 + j < 12) {
                int digit = (val >> (28 - j * 4)) & 0xF;
                hex_hash[i * 4 + j] = (digit < 10) ? ('0' + digit) : ('a' + digit - 10);
            }
        }
    }
    hex_hash[12] = '\0';
    
    // Convert hex string to integer
    uint64_t hash_value = 0;
    for (int i = 0; i < 12; i++) {
        hash_value = hash_value * 16;
        if (hex_hash[i] >= '0' && hex_hash[i] <= '9') {
            hash_value += hex_hash[i] - '0';
        } else {
            hash_value += hex_hash[i] - 'a' + 10;
        }
    }
    
    // Check if valid
    if (hash_value <= work_target && atomicCAS(found, 0, 1) == 0) {
        *result_nonce = nonce;
    }
}
"""

def generate_address():
    """Generate a new Bckn address"""
    import secrets
    private_key = secrets.token_urlsafe(32)
    
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": private_key},
                           verify=False,
                           timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n=== NEW BCKN ADDRESS GENERATED ===")
        print(f"Private Key: {private_key}")
        print(f"Address: {data['address']}")
        print(f"Save your private key securely!")
        print(f"===================================\n")
        return private_key, data['address']
    else:
        print(f"Error generating address: {response.text}")
        return None, None

def get_mining_info():
    """Get current work and last block info"""
    try:
        # Get work from simple endpoint
        work_response = requests.get(f"{BCKN_NODE}/work", verify=False, timeout=5)
        if work_response.status_code != 200:
            print(f"Work API error: {work_response.status_code}")
            return None, None
            
        work_data = work_response.json()
        if 'work' in work_data:
            work = work_data['work']
        else:
            print(f"Unexpected work format: {work_data}")
            return None, None
        
        # Get last block
        block_response = requests.get(f"{BCKN_NODE}/blocks/last", verify=False, timeout=5)
        block_data = block_response.json()
        
        # Handle genesis block case
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
            print(f"\n❌ Submission failed: {response.text}")
    except Exception as e:
        print(f"\n❌ Error submitting block: {e}")
    
    return False

def mine_gpu(address, last_hash, work, start_nonce=0, max_iterations=None):
    """GPU mining function with better nonce management"""
    global total_hashes, last_work_check
    
    # Compile CUDA kernel
    mod = SourceModule(cuda_code)
    mine_kernel = mod.get_function("mine_kernel")
    
    # Prepare prefix
    prefix = address + last_hash
    prefix_bytes = prefix.encode('ascii')
    
    # Allocate GPU memory
    d_prefix = cuda.mem_alloc(len(prefix_bytes))
    d_result_nonce = cuda.mem_alloc(8)  # uint64_t
    d_found = cuda.mem_alloc(4)  # int
    
    # Copy data to GPU
    cuda.memcpy_htod(d_prefix, prefix_bytes)
    cuda.memcpy_htod(d_found, np.int32(0))
    
    # Mining loop
    nonce_start = start_nonce
    threads_per_block = 256
    blocks_per_grid = GPU_BATCH_SIZE // threads_per_block
    iterations = 0
    
    while True:
        # Check if we should fetch new work (every 10 seconds)
        if time.time() - last_work_check > 10:
            return None, nonce_start, True  # Signal to check for new work
        
        # Check if we've reached max iterations
        if max_iterations and iterations >= max_iterations:
            return None, nonce_start, False
        
        # Reset found flag
        cuda.memcpy_htod(d_found, np.int32(0))
        
        # Launch kernel
        mine_kernel(
            d_prefix, np.int32(len(prefix_bytes)),
            np.uint64(nonce_start), np.uint64(work),
            d_result_nonce, d_found,
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_grid, 1)
        )
        
        # Check if found
        found = np.zeros(1, dtype=np.int32)
        cuda.memcpy_dtoh(found, d_found)
        found = found[0]
        
        if found:
            result_nonce = np.zeros(1, dtype=np.uint64)
            cuda.memcpy_dtoh(result_nonce, d_result_nonce)
            return int(result_nonce[0]), nonce_start + GPU_BATCH_SIZE, False
        
        # Update counters
        nonce_start += GPU_BATCH_SIZE
        total_hashes += GPU_BATCH_SIZE
        iterations += 1
        
        # Print stats
        elapsed = time.time() - start_time
        hashrate = total_hashes / elapsed if elapsed > 0 else 0
        print(f"\r[GPU] Hashrate: {hashrate/1_000_000:.2f} MH/s | "
              f"Total: {total_hashes/1_000_000_000:.2f}B | "
              f"Current: {nonce_start} | "
              f"Blocks: {blocks_found}", end='', flush=True)

def main():
    global ADDRESS, PRIVATE_KEY, total_hashes, blocks_found, last_work_check
    
    print("=== Bckn GPU Miner for NVIDIA H100 (Fixed) ===")
    print(f"Detected {NUM_GPUS} GPU(s)")
    
    # Check for existing credentials or generate new
    if len(sys.argv) > 1:
        PRIVATE_KEY = sys.argv[1]
        response = requests.post(f"{BCKN_NODE}/login", 
                               json={"privatekey": PRIVATE_KEY},
                               verify=False,
                               timeout=10)
        if response.status_code == 200:
            ADDRESS = response.json()['address']
            print(f"Mining with address: {ADDRESS}")
        else:
            print("Invalid private key!")
            return
    else:
        print("\nNo private key provided. Generate new address? (y/n)")
        if input().lower() == 'y':
            PRIVATE_KEY, ADDRESS = generate_address()
            if not PRIVATE_KEY:
                return
        else:
            print("Usage: python3 bckn-miner-gpu.py <private_key>")
            return
    
    print("\nStarting GPU mining...\n")
    
    current_work = None
    current_hash = None
    nonce_offset = 0
    
    while True:
        # Get current mining parameters
        work, last_hash = get_mining_info()
        if not work:
            print("Failed to get mining info, retrying...")
            time.sleep(5)
            continue
        
        # Check if work changed
        if work != current_work or last_hash != current_hash:
            print(f"\nNew work: {work} | Last block: {last_hash}")
            current_work = work
            current_hash = last_hash
            nonce_offset = 0  # Reset nonce when work changes
            last_work_check = time.time()
        
        # Mine on GPU with current nonce offset
        result = mine_gpu(ADDRESS, last_hash, work, nonce_offset)
        
        if result:
            nonce, next_offset, should_check_work = result
            nonce_offset = next_offset
            
            if nonce:
                print(f"\n💎 Found potential solution! Nonce: {nonce}")
                print("   Submitting to network...")
                
                # Submit solution
                if submit_block(ADDRESS, nonce):
                    blocks_found += 1
                    # Reset nonce offset after successful block
                    nonce_offset = 0
                else:
                    print("   Submission failed, continuing mining...")
            
            if should_check_work:
                last_work_check = time.time()
                # Continue to next iteration to check for new work

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nMining stopped.")
        print(f"Total hashes: {total_hashes:,}")
        print(f"Blocks found: {blocks_found}")
        elapsed = time.time() - start_time
        print(f"Average hashrate: {total_hashes/elapsed/1_000_000:.2f} MH/s")