#!/bin/bash

# Quick fix for GPU power limit issues

echo "Fixing GPU power limit settings..."

# First, enable persistence mode (required for power management)
echo "Enabling GPU persistence mode..."
sudo nvidia-smi -pm 1

# On some systems, we need to load the nvidia kernel module with proper permissions
if ! lsmod | grep -q nvidia; then
    echo "Loading NVIDIA kernel modules..."
    sudo modprobe nvidia
    sudo modprobe nvidia_modeset
    sudo modprobe nvidia_uvm
fi

# Detect GPU model
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"

# Check if we can query power limits first
echo "Checking available power limit range..."
POWER_INFO=$(nvidia-smi -q -d POWER | grep -A 2 "Power Limit")

# Set appropriate power limit
if [[ $GPU_NAME == *"RTX 5090"* ]] || [[ $GPU_NAME == *"RTX 4090"* ]]; then
    echo "Setting power limit to 500W for RTX 5090/4090..."
    sudo nvidia-smi -pl 500 2>/dev/null || {
        echo "Note: Power limit setting may not be supported in this environment."
        echo "This is common in containerized/virtualized environments."
        echo "The GPU will use default power settings."
    }
elif [[ $GPU_NAME == *"H100"* ]]; then
    echo "Setting power limit to 350W for H100..."
    sudo nvidia-smi -pl 350 2>/dev/null || echo "Note: Using default power settings."
elif [[ $GPU_NAME == *"RTX 4080"* ]]; then
    echo "Setting power limit to 320W for RTX 4080..."
    sudo nvidia-smi -pl 320 2>/dev/null || echo "Note: Using default power settings."
elif [[ $GPU_NAME == *"RTX 3090"* ]]; then
    echo "Setting power limit to 350W for RTX 3090..."
    sudo nvidia-smi -pl 350 2>/dev/null || echo "Note: Using default power settings."
elif [[ $GPU_NAME == *"A100"* ]]; then
    echo "Setting power limit to 300W for A100..."
    sudo nvidia-smi -pl 300 2>/dev/null || echo "Note: Using default power settings."
else
    echo "Unknown GPU model. You may need to set power limits manually."
    echo "To find valid power limits for your GPU, run:"
    echo "nvidia-smi -q -d POWER"
fi

echo
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,power.draw,power.limit,temperature.gpu --format=csv
echo
echo "Note: If power limit settings failed, this is often due to:"
echo "1. Running in a container/VM without full GPU access"
echo "2. GPU driver restrictions"
echo "3. Hardware limitations"
echo ""
echo "The miner will still work fine with default power settings!"