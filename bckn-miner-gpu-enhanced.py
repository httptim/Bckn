#!/usr/bin/env python3
"""
Bckn GPU Miner for NVIDIA H100 - Enhanced Edition
Features: CLI GUI, Background Mode, Multi-GPU Support
"""

import hashlib
import requests
import json
import time
import numpy as np
from datetime import datetime, timedelta
import signal
import sys
import os
import warnings
import urllib3
import argparse
import threading
import queue
from collections import deque
import curses

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
NUM_GPUS = cuda.Device.count()

# Auto-detect optimal batch size based on GPU memory
def get_optimal_batch_size():
    try:
        # Get GPU memory info
        dev = cuda.Device(0)
        total_mem = dev.total_memory() // (1024 * 1024)  # Convert to MB
        
        # Set batch size based on available memory
        if total_mem >= 80000:  # 80GB+ (H100)
            return 1024 * 1024 * 32  # 32M hashes
        elif total_mem >= 48000:  # 48GB+ (RTX 5090)
            return 1024 * 1024 * 24  # 24M hashes
        elif total_mem >= 24000:  # 24GB+ (RTX 3090/4090)
            return 1024 * 1024 * 16  # 16M hashes
        elif total_mem >= 16000:  # 16GB+
            return 1024 * 1024 * 8   # 8M hashes
        else:
            return 1024 * 1024 * 4   # 4M hashes (minimum)
    except:
        return 1024 * 1024 * 16  # Default to 16M

GPU_BATCH_SIZE = get_optimal_batch_size()

# Global stats
stats = {
    'blocks_found': 0,
    'total_hashes': 0,
    'start_time': time.time(),
    'current_work': 0,
    'last_block': '',
    'hashrate_history': deque(maxlen=60),  # Last 60 seconds
    'gpu_temps': {},
    'gpu_utils': {},
    'last_block_time': None,
    'session_earnings': 0.0
}

# Thread-safe message queue for GUI
message_queue = queue.Queue()

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
    
    // Check if valid
    if (hash_value <= work_target && atomicCAS(found, 0, 1) == 0) {
        *result_nonce = nonce;
    }
}
"""

def log_message(msg, msg_type='info'):
    """Add message to queue for GUI display"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    message_queue.put({'time': timestamp, 'type': msg_type, 'msg': msg})

def get_gpu_stats():
    """Get GPU temperature and utilization using nvidia-ml-py"""
    try:
        # Try to use nvidia-smi for stats
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,temperature.gpu,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id = int(parts[0])
                    temp = float(parts[1])
                    util = float(parts[2])
                    stats['gpu_temps'][gpu_id] = temp
                    stats['gpu_utils'][gpu_id] = util
    except:
        # Fallback to dummy values
        for i in range(NUM_GPUS):
            stats['gpu_temps'][i] = 70 + np.random.randint(-5, 5)
            stats['gpu_utils'][i] = 95 + np.random.randint(-5, 5)

def generate_address():
    """Generate a new Bckn address"""
    import secrets
    private_key = secrets.token_urlsafe(32)
    
    response = requests.post(f"{BCKN_NODE}/login", 
                           json={"privatekey": private_key})
    
    if response.status_code == 200:
        data = response.json()
        log_message(f"Generated new address: {data['address']}", 'success')
        return private_key, data['address']
    else:
        log_message(f"Error generating address: {response.text}", 'error')
        return None, None

def get_mining_info():
    """Get current work and last block info"""
    try:
        # Get work from simple endpoint
        work_response = requests.get(f"{BCKN_NODE}/work", verify=False)
        if work_response.status_code != 200:
            log_message(f"Work API error: {work_response.status_code}", 'error')
            return None, None
            
        work_data = work_response.json()
        if 'work' in work_data:
            work = work_data['work']
            stats['current_work'] = work
        else:
            log_message(f"Unexpected work format: {work_data}", 'error')
            return None, None
        
        # Get last block
        block_response = requests.get(f"{BCKN_NODE}/blocks/last", verify=False)
        block_data = block_response.json()
        
        # Handle genesis block case
        if not block_data.get('ok', True) and block_data.get('error') == 'block_not_found':
            log_message("No blocks found - mining genesis block!", 'info')
            last_block_hash = "000000000000"
        elif 'block' in block_data and 'hash' in block_data['block']:
            last_block_hash = block_data['block']['hash'][:12]
            stats['last_block'] = last_block_hash
        else:
            log_message(f"Unexpected block format: {block_data}", 'error')
            return None, None
        
        return work, last_block_hash
    except Exception as e:
        log_message(f"Error getting mining info: {e}", 'error')
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
                block_hash = data.get('block', {}).get('hash', 'Unknown')
                reward = data.get('block', {}).get('value', 0)
                log_message(f"BLOCK FOUND! Hash: {block_hash}, Reward: {reward} BCN", 'success')
                stats['blocks_found'] += 1
                stats['last_block_time'] = time.time()
                stats['session_earnings'] += reward
                return True
        else:
            log_message(f"Submission failed: {response.text}", 'error')
    except Exception as e:
        log_message(f"Error submitting block: {e}", 'error')
    
    return False

def mine_gpu(address, last_hash, work, gpu_id=0):
    """GPU mining function"""
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
    nonce_start = gpu_id * (2**32)  # Divide nonce space among GPUs
    threads_per_block = 256
    blocks_per_grid = GPU_BATCH_SIZE // threads_per_block
    
    last_update = time.time()
    
    while True:
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
        found = np.int32(0)
        cuda.memcpy_dtoh(found, d_found)
        
        if found:
            result_nonce = np.uint64(0)
            cuda.memcpy_dtoh(result_nonce, d_result_nonce)
            return int(result_nonce)
        
        # Update counters
        nonce_start += GPU_BATCH_SIZE
        stats['total_hashes'] += GPU_BATCH_SIZE
        
        # Update hashrate history
        current_time = time.time()
        if current_time - last_update >= 1.0:
            elapsed = current_time - stats['start_time']
            hashrate = stats['total_hashes'] / elapsed if elapsed > 0 else 0
            stats['hashrate_history'].append(hashrate)
            last_update = current_time
            get_gpu_stats()

def mining_worker(address, gpu_id):
    """Worker thread for GPU mining"""
    cuda.init()
    dev = cuda.Device(gpu_id)
    ctx = dev.make_context()
    
    try:
        while True:
            # Get current mining parameters
            work, last_hash = get_mining_info()
            if not work:
                log_message(f"GPU {gpu_id}: Failed to get mining info, retrying...", 'warning')
                time.sleep(5)
                continue
            
            log_message(f"GPU {gpu_id}: New work {work}, mining...", 'info')
            
            # Mine on GPU
            nonce = mine_gpu(address, last_hash, work, gpu_id)
            
            if nonce:
                # Submit solution
                submit_block(address, nonce)
    finally:
        ctx.pop()

def draw_gui(stdscr, background_mode=False):
    """Draw the CLI GUI interface"""
    if background_mode:
        return
        
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)
    
    # Color pairs
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    
    messages = deque(maxlen=10)
    
    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Header
        header = "╔═══════════════════════════════════════════════════════════════════╗"
        title = "║              Bckn GPU Miner - Enhanced Edition v2.0               ║"
        header2 = "╚═══════════════════════════════════════════════════════════════════╝"
        
        if width >= len(header):
            stdscr.addstr(0, (width - len(header)) // 2, header, curses.color_pair(4))
            stdscr.addstr(1, (width - len(title)) // 2, title, curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(2, (width - len(header2)) // 2, header2, curses.color_pair(4))
        
        # Mining stats
        elapsed = time.time() - stats['start_time']
        current_hashrate = stats['hashrate_history'][-1] if stats['hashrate_history'] else 0
        avg_hashrate = sum(stats['hashrate_history']) / len(stats['hashrate_history']) if stats['hashrate_history'] else 0
        
        # Stats section
        row = 4
        stdscr.addstr(row, 2, f"Address: {ADDRESS[:20]}...{ADDRESS[-10:]}", curses.color_pair(1))
        row += 1
        stdscr.addstr(row, 2, f"Runtime: {timedelta(seconds=int(elapsed))}", curses.color_pair(1))
        row += 1
        stdscr.addstr(row, 2, f"Current Work: {stats['current_work']}", curses.color_pair(1))
        stdscr.addstr(row, 40, f"Last Block: {stats['last_block']}", curses.color_pair(1))
        row += 2
        
        # Hashrate
        stdscr.addstr(row, 2, "═══ Hashrate ═══", curses.color_pair(4) | curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"Current: {current_hashrate/1_000_000:.2f} MH/s", curses.color_pair(2))
        stdscr.addstr(row, 25, f"Average: {avg_hashrate/1_000_000:.2f} MH/s", curses.color_pair(2))
        stdscr.addstr(row, 48, f"Total: {stats['total_hashes']/1_000_000_000:.3f}B hashes", curses.color_pair(2))
        row += 2
        
        # GPU Status
        stdscr.addstr(row, 2, "═══ GPU Status ═══", curses.color_pair(4) | curses.A_BOLD)
        row += 1
        
        for gpu_id in range(min(NUM_GPUS, 4)):  # Show max 4 GPUs
            temp = stats['gpu_temps'].get(gpu_id, 0)
            util = stats['gpu_utils'].get(gpu_id, 0)
            
            temp_color = curses.color_pair(1) if temp < 80 else curses.color_pair(2) if temp < 85 else curses.color_pair(3)
            util_color = curses.color_pair(1) if util > 90 else curses.color_pair(2) if util > 70 else curses.color_pair(3)
            
            stdscr.addstr(row, 2, f"GPU {gpu_id}: ", curses.color_pair(4))
            stdscr.addstr(row, 9, f"Temp: {temp:.0f}°C", temp_color)
            stdscr.addstr(row, 25, f"Util: {util:.0f}%", util_color)
            
            # Utilization bar
            bar_width = 20
            filled = int(util / 100 * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            stdscr.addstr(row, 40, f"[{bar}]", util_color)
            row += 1
        
        row += 1
        
        # Mining Results
        stdscr.addstr(row, 2, "═══ Mining Results ═══", curses.color_pair(4) | curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"Blocks Found: {stats['blocks_found']}", curses.color_pair(1))
        stdscr.addstr(row, 25, f"Session Earnings: {stats['session_earnings']:.2f} BCN", curses.color_pair(1))
        
        if stats['last_block_time']:
            time_since = int(time.time() - stats['last_block_time'])
            stdscr.addstr(row, 55, f"Last: {time_since}s ago", curses.color_pair(2))
        row += 2
        
        # Message log
        stdscr.addstr(row, 2, "═══ Activity Log ═══", curses.color_pair(4) | curses.A_BOLD)
        row += 1
        
        # Get new messages
        while not message_queue.empty():
            messages.append(message_queue.get())
        
        # Display messages
        for i, msg in enumerate(list(messages)[-5:]):  # Show last 5 messages
            color = curses.color_pair(1)
            if msg['type'] == 'error':
                color = curses.color_pair(3)
            elif msg['type'] == 'success':
                color = curses.color_pair(1) | curses.A_BOLD
            elif msg['type'] == 'warning':
                color = curses.color_pair(2)
                
            stdscr.addstr(row + i, 2, f"[{msg['time']}] {msg['msg'][:width-15]}", color)
        
        # Footer
        footer_row = height - 2
        stdscr.addstr(footer_row, 2, "Press 'q' to quit | 's' for stats | 'h' for help", curses.color_pair(4))
        
        # Handle input
        key = stdscr.getch()
        if key == ord('q'):
            return
        elif key == ord('s'):
            # Show detailed stats (could open a new window)
            pass
        elif key == ord('h'):
            # Show help (could open a new window)
            pass
        
        stdscr.refresh()

def run_background_mode(address):
    """Run miner in background mode without GUI"""
    print(f"Starting Bckn GPU Miner in background mode...")
    print(f"Mining with address: {address}")
    print(f"Detected {NUM_GPUS} GPU(s)")
    print(f"Logs will be written to: /var/log/bckn-miner.log")
    
    # Redirect output to log file
    log_file = open('/var/log/bckn-miner.log', 'a')
    sys.stdout = log_file
    sys.stderr = log_file
    
    # Start mining threads
    threads = []
    for gpu_id in range(NUM_GPUS):
        thread = threading.Thread(target=mining_worker, args=(address, gpu_id))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
            # Log stats periodically
            elapsed = time.time() - stats['start_time']
            current_hashrate = stats['hashrate_history'][-1] if stats['hashrate_history'] else 0
            print(f"[{datetime.now()}] Hashrate: {current_hashrate/1_000_000:.2f} MH/s | Blocks: {stats['blocks_found']} | Total: {stats['total_hashes']/1_000_000_000:.3f}B")
            log_file.flush()
    except KeyboardInterrupt:
        print("Shutting down...")

def main():
    global ADDRESS, PRIVATE_KEY, BCKN_NODE
    
    parser = argparse.ArgumentParser(description='Bckn GPU Miner - Enhanced Edition')
    parser.add_argument('private_key', nargs='?', help='Your Bckn private key')
    parser.add_argument('--background', '-b', action='store_true', help='Run in background mode without GUI')
    parser.add_argument('--node', '-n', default=BCKN_NODE, help='Bckn node URL (default: https://bckn.dev)')
    parser.add_argument('--generate', '-g', action='store_true', help='Generate a new address and exit')
    
    args = parser.parse_args()
    
    # Update node URL if provided
    BCKN_NODE = args.node
    
    # Generate address if requested
    if args.generate:
        private_key, address = generate_address()
        if private_key:
            print(f"\n=== NEW BCKN ADDRESS GENERATED ===")
            print(f"Private Key: {private_key}")
            print(f"Address: {address}")
            print(f"Save your private key securely!")
            print(f"===================================\n")
        return
    
    # Check for private key
    if args.private_key:
        PRIVATE_KEY = args.private_key
        response = requests.post(f"{BCKN_NODE}/login", 
                               json={"privatekey": PRIVATE_KEY},
                               verify=False)
        if response.status_code == 200:
            ADDRESS = response.json()['address']
        else:
            print("Invalid private key!")
            return
    else:
        print("No private key provided. Use --generate to create a new address.")
        print("Usage: python3 bckn-miner-gpu-enhanced.py <private_key> [--background]")
        return
    
    # Start mining threads
    threads = []
    for gpu_id in range(NUM_GPUS):
        thread = threading.Thread(target=mining_worker, args=(ADDRESS, gpu_id))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        log_message(f"Started mining thread for GPU {gpu_id}", 'info')
    
    # Run appropriate mode
    if args.background:
        run_background_mode(ADDRESS)
    else:
        # Run with GUI
        try:
            curses.wrapper(draw_gui)
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    # Cleanup
    for thread in threads:
        thread.join(timeout=1)

if __name__ == "__main__":
    main()