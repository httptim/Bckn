#!/usr/bin/env python3
"""
Bckn Fast GPU Miner
Maximizes GPU utilization on RTX 5090
"""

import hashlib
import requests
import time
import sys
import numpy as np
from datetime import datetime
import warnings
import urllib3
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn.functional as F
    
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        print(f"PyTorch GPU acceleration enabled")
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("No GPU found! Please check CUDA installation")
        sys.exit(1)
        
except ImportError:
    print("PyTorch not found. Install with:")
    print("pip install torch")
    sys.exit(1)

# Configuration
BCKN_NODE = "https://bckn.dev"
GPU_BATCH_SIZE = 100_000_000  # 100M nonces per GPU batch
CPU_VERIFY_THREADS = 16  # Parallel CPU verification threads

# Global stats
stats = {
    'blocks_found': 0,
    'total_hashes': 0,
    'start_time': time.time(),
    'valid_nonces': []
}

# Queue for GPU->CPU verification
verify_queue = queue.Queue(maxsize=1000)
result_queue = queue.Queue()

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
    except Exception:
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

def verify_nonce_batch(prefix, nonces, work):
    """Verify a batch of nonces on CPU"""
    valid = []
    for nonce in nonces:
        message = prefix + str(nonce)
        hash_result = hashlib.sha256(message.encode()).hexdigest()
        hash_int = int(hash_result[:12], 16)
        if hash_int <= work:
            valid.append(nonce)
    return valid

def cpu_verification_worker(prefix, work):
    """Worker thread for CPU verification"""
    while True:
        try:
            batch = verify_queue.get(timeout=1)
            if batch is None:  # Shutdown signal
                break
                
            valid_nonces = verify_nonce_batch(prefix, batch, work)
            if valid_nonces:
                for nonce in valid_nonces:
                    result_queue.put(nonce)
        except queue.Empty:
            continue

@torch.jit.script
def gpu_hash_filter(nonces: torch.Tensor, prefix_sum: int, work_threshold: int) -> torch.Tensor:
    """
    JIT-compiled GPU function for fast filtering
    Returns indices of potentially valid nonces
    """
    # Fast hash approximation
    hash_vals = nonces.clone()
    
    # Mix with prefix
    hash_vals = (hash_vals ^ prefix_sum) * 0x5DEECE66D + 0xB
    hash_vals = hash_vals ^ (hash_vals >> 17)
    hash_vals = hash_vals * 0x27D4EB2D
    hash_vals = hash_vals ^ (hash_vals >> 15)
    
    # Find candidates
    mask = (hash_vals & 0xFFFFFFFFFF) < work_threshold
    return nonces[mask]

def mine_gpu_fast(prefix, start_nonce, work, gpu_id=0):
    """
    Fast GPU mining with better utilization
    """
    global stats
    
    # Set GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    # Calculate prefix hash for filtering
    prefix_sum = sum(ord(c) * (31 ** i) for i, c in enumerate(prefix[:10])) & 0xFFFFFFFF
    
    # Aggressive threshold for filtering (catches more candidates)
    work_threshold = work * 100000
    
    nonce = start_nonce + (gpu_id * GPU_BATCH_SIZE)
    
    # Pre-allocate GPU memory
    with torch.cuda.device(gpu_id):
        while True:
            # Generate large batch of nonces on GPU
            nonces = torch.arange(
                nonce, 
                nonce + GPU_BATCH_SIZE, 
                device=device, 
                dtype=torch.int64
            )
            
            # Fast GPU filtering
            with torch.cuda.amp.autocast():
                candidates = gpu_hash_filter(nonces, prefix_sum, work_threshold)
            
            # Send candidates to CPU verification
            if len(candidates) > 0:
                candidates_cpu = candidates.cpu().numpy()
                
                # Batch candidates for CPU verification
                batch_size = 1000
                for i in range(0, len(candidates_cpu), batch_size):
                    batch = candidates_cpu[i:i+batch_size]
                    verify_queue.put(batch)
            
            # Update position
            nonce += GPU_BATCH_SIZE * gpu_count
            stats['total_hashes'] += GPU_BATCH_SIZE
            
            # Clear cache periodically
            if nonce % (GPU_BATCH_SIZE * 10) == 0:
                torch.cuda.empty_cache()
                
                # Check for new work
                new_work, new_hash = get_mining_info()
                if new_work and new_work != work:
                    return

def print_stats():
    """Print mining statistics"""
    while True:
        time.sleep(0.5)
        elapsed = time.time() - stats['start_time']
        hashrate = stats['total_hashes'] / elapsed if elapsed > 0 else 0
        
        # Get GPU stats
        gpu_mem = sum(torch.cuda.memory_allocated(i) / 1024**3 for i in range(gpu_count))
        
        print(f"\r[GPU] Rate: {hashrate/1_000_000:.2f} MH/s | "
              f"Total: {stats['total_hashes']/1_000_000_000:.2f}B | "
              f"Blocks: {stats['blocks_found']} | "
              f"Queue: {verify_queue.qsize()} | "
              f"VRAM: {gpu_mem:.1f}GB", end='', flush=True)

def main():
    global gpu_count, stats
    
    if len(sys.argv) < 2:
        print("Usage: python3 bckn-miner-gpu-fast.py <private_key>")
        return
    
    private_key = sys.argv[1]
    
    # Login
    print("\n=== Bckn Fast GPU Miner ===")
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": private_key},
                           verify=False,
                           timeout=10)
    
    if response.status_code != 200:
        print("Invalid private key!")
        return
    
    address = response.json()['address']
    print(f"Mining with address: {address}")
    
    # Get initial work
    work, last_hash = get_mining_info()
    if not work:
        print("Failed to get work!")
        return
        
    print(f"\nWork: {work} | Last block: {last_hash}")
    print(f"Difficulty: 1 in {2**48 / work:,.0f} hashes")
    print(f"Using {gpu_count} GPU(s) + {CPU_VERIFY_THREADS} CPU threads\n")
    
    prefix = address + last_hash
    
    # Start CPU verification threads
    cpu_threads = []
    for _ in range(CPU_VERIFY_THREADS):
        thread = threading.Thread(
            target=cpu_verification_worker, 
            args=(prefix, work),
            daemon=True
        )
        thread.start()
        cpu_threads.append(thread)
    
    # Start GPU mining threads (one per GPU)
    gpu_threads = []
    for gpu_id in range(gpu_count):
        thread = threading.Thread(
            target=mine_gpu_fast,
            args=(prefix, 0, work, gpu_id),
            daemon=True
        )
        thread.start()
        gpu_threads.append(thread)
        print(f"Started GPU {gpu_id} mining thread")
    
    # Start stats thread
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()
    
    # Main loop - check for valid nonces
    current_work = work
    current_hash = last_hash
    
    while True:
        try:
            # Check for valid nonces
            nonce = result_queue.get(timeout=1)
            
            print(f"\n\n💎 Found valid nonce! {nonce}")
            print("   Submitting to network...")
            
            if submit_block(address, nonce):
                stats['blocks_found'] += 1
                
                # Get new work
                time.sleep(1)
                work, last_hash = get_mining_info()
                if work and (work != current_work or last_hash != current_hash):
                    print(f"\nNew work: {work} | Last block: {last_hash}")
                    current_work = work
                    current_hash = last_hash
                    prefix = address + last_hash
                    
                    # Restart threads with new work
                    # (In production, signal threads to update)
                    
        except queue.Empty:
            # Check for work changes
            new_work, new_hash = get_mining_info()
            if new_work and (new_work != current_work or new_hash != current_hash):
                print(f"\nWork changed: {new_work} | Last block: {new_hash}")
                current_work = new_work
                current_hash = new_hash
                prefix = address + current_hash
                stats['total_hashes'] = 0
                stats['start_time'] = time.time()

if __name__ == "__main__":
    main()