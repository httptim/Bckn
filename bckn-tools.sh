#!/bin/bash

# Bckn Tools - Utility script for Bckn operations

BCKN_NODE="https://bckn.dev"

# Function to generate new address
generate_address() {
    echo "=== Generate New Bckn Address ==="
    
    # Generate random private key (32 chars)
    PRIVATE_KEY=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)
    
    # Get address from API
    RESPONSE=$(curl -s -X POST "${BCKN_NODE}/login" \
        -H "Content-Type: application/json" \
        -d "{\"privatekey\": \"${PRIVATE_KEY}\"}")
    
    ADDRESS=$(echo "$RESPONSE" | grep -o '"address":"[^"]*' | cut -d'"' -f4)
    
    if [ -n "$ADDRESS" ]; then
        echo ""
        echo "✅ New Bckn Address Generated!"
        echo "================================"
        echo "Private Key: ${PRIVATE_KEY}"
        echo "Address:     ${ADDRESS}"
        echo "================================"
        echo ""
        echo "⚠️  IMPORTANT: Save your private key securely!"
        echo "    You'll need it to access your funds."
    else
        echo "❌ Error generating address"
        echo "Response: $RESPONSE"
    fi
}

# Function to check balance
check_balance() {
    if [ -z "$1" ]; then
        echo "Usage: $0 balance <address>"
        return
    fi
    
    RESPONSE=$(curl -s "${BCKN_NODE}/addresses/${1}")
    BALANCE=$(echo "$RESPONSE" | grep -o '"balance":[0-9]*' | cut -d: -f2)
    
    if [ -n "$BALANCE" ]; then
        echo "Address: $1"
        echo "Balance: ${BALANCE} BCN"
    else
        echo "Error checking balance"
    fi
}

# Function to get current work
get_work() {
    RESPONSE=$(curl -s "${BCKN_NODE}/work/detailed")
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
}

# Function to get network info
network_info() {
    echo "=== Bckn Network Info ==="
    
    # Get supply
    SUPPLY=$(curl -s "${BCKN_NODE}/supply")
    echo "Supply: $SUPPLY"
    
    # Get last block
    BLOCK=$(curl -s "${BCKN_NODE}/blocks/last" | jq -r '.block.height' 2>/dev/null)
    echo "Latest Block: $BLOCK"
    
    # Get work
    WORK=$(curl -s "${BCKN_NODE}/work")
    echo "Current Work: $WORK"
    
    # Get MOTD
    MOTD=$(curl -s "${BCKN_NODE}/motd")
    echo "MOTD: $MOTD"
}

# Function to start CPU mining on Mac
start_mining_mac() {
    echo "=== Starting Bckn Mining on macOS ==="
    
    # Check if Python 3 is installed
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required. Install with: brew install python3"
        exit 1
    fi
    
    # Install required packages
    echo "Installing required Python packages..."
    pip3 install requests
    
    # Download miner if not exists
    if [ ! -f "bckn-miner-mac.py" ]; then
        echo "Downloading Bckn miner..."
        curl -O https://raw.githubusercontent.com/httptim/Bckn/master/bckn-miner-mac.py
        chmod +x bckn-miner-mac.py
    fi
    
    # Run miner
    if [ -n "$1" ]; then
        python3 bckn-miner-mac.py "$1"
    else
        python3 bckn-miner-mac.py
    fi
}

# Function to setup GPU mining on Ubuntu
setup_gpu_mining() {
    echo "=== Setting up GPU Mining on Ubuntu with NVIDIA GPUs ==="
    
    # This should be run on the GPU droplet
    
    # Update system
    sudo apt update && sudo apt upgrade -y
    
    # Install NVIDIA drivers and CUDA
    echo "Installing NVIDIA drivers and CUDA..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda
    
    # Install Python and dependencies
    sudo apt install -y python3 python3-pip
    pip3 install requests numpy cupy-cuda12x pycuda
    
    # Download GPU miner
    wget https://raw.githubusercontent.com/httptim/Bckn/master/bckn-miner-gpu.py
    chmod +x bckn-miner-gpu.py
    
    echo "✅ GPU mining setup complete!"
    echo "Run with: python3 bckn-miner-gpu.py <private_key>"
}

# Main menu
case "$1" in
    generate)
        generate_address
        ;;
    balance)
        check_balance "$2"
        ;;
    work)
        get_work
        ;;
    info)
        network_info
        ;;
    mine)
        start_mining_mac "$2"
        ;;
    setup-gpu)
        setup_gpu_mining
        ;;
    *)
        echo "Bckn Tools - Utility script for Bckn operations"
        echo ""
        echo "Usage: $0 {command} [options]"
        echo ""
        echo "Commands:"
        echo "  generate         - Generate new Bckn address"
        echo "  balance <addr>   - Check balance of address"
        echo "  work            - Get current mining work"
        echo "  info            - Get network information"
        echo "  mine [key]      - Start CPU mining on Mac"
        echo "  setup-gpu       - Setup GPU mining on Ubuntu"
        echo ""
        echo "Examples:"
        echo "  $0 generate"
        echo "  $0 balance k5ztameslf"
        echo "  $0 mine your_private_key_here"
        ;;
esac