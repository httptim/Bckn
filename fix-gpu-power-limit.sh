#!/bin/bash

# Quick fix for GPU power limit issues

echo "Fixing GPU power limit settings..."

# Detect GPU model
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"

# Set appropriate power limit
if [[ $GPU_NAME == *"RTX 5090"* ]] || [[ $GPU_NAME == *"RTX 4090"* ]]; then
    echo "Setting power limit to 500W for RTX 5090/4090..."
    nvidia-smi -pl 500
elif [[ $GPU_NAME == *"H100"* ]]; then
    echo "Setting power limit to 350W for H100..."
    nvidia-smi -pl 350
elif [[ $GPU_NAME == *"RTX 4080"* ]]; then
    echo "Setting power limit to 320W for RTX 4080..."
    nvidia-smi -pl 320
elif [[ $GPU_NAME == *"RTX 3090"* ]]; then
    echo "Setting power limit to 350W for RTX 3090..."
    nvidia-smi -pl 350
elif [[ $GPU_NAME == *"A100"* ]]; then
    echo "Setting power limit to 300W for A100..."
    nvidia-smi -pl 300
else
    echo "Unknown GPU model. You may need to set power limits manually."
    echo "To find valid power limits for your GPU, run:"
    echo "nvidia-smi -q -d POWER"
fi

echo "Done! Power limits updated."