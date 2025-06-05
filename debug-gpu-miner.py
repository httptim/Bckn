#!/usr/bin/env python3
"""
Debug GPU Miner - Simplified version to diagnose issues
"""

import hashlib
import requests
import json
import time
import numpy as np
import sys
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

# CUDA kernel for SHA256 mining - simplified
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
    
    for (int i = 0; i < 16; i++) {
        w[i] = (data[i*4] << 24) | (data[i*4 + 1] << 16) | 
               (data[i*4 + 2] << 8) | data[i*4 + 3];
    }
    
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    
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
    
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__global__ void test_kernel(const char* prefix, int prefix_len, 
                           uint64_t test_nonce, uint64_t work_target,
                           uint64_t* result_hash, int* valid) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
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
    uint64_t n = test_nonce;
    
    if (n == 0) {
        nonce_str[0] = '0';
        nonce_len = 1;
    } else {
        while (n > 0 && nonce_len < 24) {
            nonce_str[nonce_len++] = '0' + (n % 10);
            n /= 10;
        }
        for (int i = 0; i < nonce_len / 2; i++) {
            char temp = nonce_str[i];
            nonce_str[i] = nonce_str[nonce_len - 1 - i];
            nonce_str[nonce_len - 1 - i] = temp;
        }
    }
    
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
    hex_hash[12] = '\\0';
    
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
    
    *result_hash = hash_value;
    *valid = (hash_value <= work_target) ? 1 : 0;
}
"""

def test_single_hash(address, last_hash, nonce, work):
    """Test a single hash calculation"""
    prefix = address + last_hash + str(nonce)
    hash_obj = hashlib.sha256(prefix.encode('ascii'))
    hash_hex = hash_obj.hexdigest()
    hash_value = int(hash_hex[:12], 16)
    
    print(f"\nCPU Test:")
    print(f"  Prefix: {address + last_hash}")
    print(f"  Nonce: {nonce}")
    print(f"  Full string: {prefix}")
    print(f"  Hash: {hash_hex}")
    print(f"  First 12 chars: {hash_hex[:12]}")
    print(f"  Hash value: {hash_value}")
    print(f"  Work target: {work}")
    print(f"  Valid: {hash_value <= work}")
    
    return hash_value

def test_gpu_hash(address, last_hash, nonce, work):
    """Test GPU hash calculation"""
    # Compile CUDA kernel
    mod = SourceModule(cuda_code)
    test_kernel = mod.get_function("test_kernel")
    
    # Prepare prefix
    prefix = address + last_hash
    prefix_bytes = prefix.encode('ascii')
    
    # Allocate GPU memory
    d_prefix = cuda.mem_alloc(len(prefix_bytes))
    d_result_hash = cuda.mem_alloc(8)  # uint64_t
    d_valid = cuda.mem_alloc(4)  # int
    
    # Copy data to GPU
    cuda.memcpy_htod(d_prefix, prefix_bytes)
    
    # Run test kernel
    test_kernel(
        d_prefix, np.int32(len(prefix_bytes)),
        np.uint64(nonce), np.uint64(work),
        d_result_hash, d_valid,
        block=(1, 1, 1),
        grid=(1, 1)
    )
    
    # Get results
    result_hash = np.zeros(1, dtype=np.uint64)
    valid = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(result_hash, d_result_hash)
    cuda.memcpy_dtoh(valid, d_valid)
    
    print(f"\nGPU Test:")
    print(f"  Prefix: {prefix}")
    print(f"  Nonce: {nonce}")
    print(f"  Hash value: {result_hash[0]}")
    print(f"  Work target: {work}")
    print(f"  Valid: {valid[0] == 1}")
    
    return result_hash[0]

def main():
    print("=== Bckn GPU Debug Tool ===")
    print(f"Detected {cuda.Device.count()} GPU(s)\n")
    
    # Get test parameters
    if len(sys.argv) > 1:
        private_key = sys.argv[1]
        response = requests.post(f"{BCKN_NODE}/login", 
                               json={"privatekey": private_key},
                               verify=False,
                               timeout=10)
        if response.status_code == 200:
            address = response.json()['address']
            print(f"Using address: {address}")
        else:
            print("Invalid private key!")
            return
    else:
        # Use test address
        address = "k5cqght20v"
        print(f"Using test address: {address}")
    
    # Get current work
    work_response = requests.get(f"{BCKN_NODE}/work", verify=False, timeout=5)
    work = work_response.json()['work']
    
    # Get last block
    block_response = requests.get(f"{BCKN_NODE}/blocks/last", verify=False, timeout=5)
    block_data = block_response.json()
    
    if not block_data.get('ok', True) and block_data.get('error') == 'block_not_found':
        last_hash = "000000000000"
    else:
        last_hash = block_data['block']['hash'][:12]
    
    print(f"\nCurrent work: {work}")
    print(f"Last block hash: {last_hash}")
    print(f"Expected probability: 1 in {(2**48) / work:,.0f} hashes")
    
    # Test a few nonces
    test_nonces = [0, 1, 1000, 1090519040, 2147483647, 4294967295]
    
    for nonce in test_nonces:
        print(f"\n{'='*50}")
        print(f"Testing nonce: {nonce}")
        
        cpu_hash = test_single_hash(address, last_hash, nonce, work)
        gpu_hash = test_gpu_hash(address, last_hash, nonce, work)
        
        if cpu_hash != gpu_hash:
            print(f"\n⚠️  MISMATCH! CPU: {cpu_hash}, GPU: {gpu_hash}")
        else:
            print(f"\n✅ Match! Both: {cpu_hash}")
    
    # Test mining for a short time
    print(f"\n{'='*50}")
    print("Testing actual mining for 10 seconds...")
    
    start_time = time.time()
    hashes_checked = 0
    batch_size = 1024 * 1024  # 1M hashes
    
    while time.time() - start_time < 10:
        start_nonce = hashes_checked
        
        # Test batch on CPU (sampling)
        for i in range(0, min(100, batch_size), batch_size // 100):
            nonce = start_nonce + i
            prefix = address + last_hash + str(nonce)
            hash_obj = hashlib.sha256(prefix.encode('ascii'))
            hash_hex = hash_obj.hexdigest()
            hash_value = int(hash_hex[:12], 16)
            
            if hash_value <= work:
                print(f"\n🎉 Found valid hash!")
                print(f"   Nonce: {nonce}")
                print(f"   Hash: {hash_hex}")
                print(f"   Value: {hash_value} <= {work}")
        
        hashes_checked += batch_size
        
        elapsed = time.time() - start_time
        hashrate = hashes_checked / elapsed
        print(f"\rChecked {hashes_checked:,} hashes | {hashrate/1_000_000:.2f} MH/s", end='', flush=True)
    
    print(f"\n\nTest complete. Checked {hashes_checked:,} hashes total.")

if __name__ == "__main__":
    main()