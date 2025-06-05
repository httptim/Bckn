# Bckn GPU Mining Setup for Digital Ocean H100

## 🚀 Quick Start

### 1. **SSH into your GPU droplet**
```bash
ssh root@your-gpu-droplet-ip
```

### 2. **Install NVIDIA drivers and CUDA** (if not already installed)
```bash
# Update system
apt update && apt upgrade -y

# Install NVIDIA drivers
apt install -y nvidia-driver-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install -y cuda-12-3

# Reboot to load drivers
reboot
```

### 3. **Verify GPU is working**
```bash
nvidia-smi
```
You should see your H100 GPU listed.

### 4. **Install Python and dependencies**
```bash
# Install Python and pip
apt install -y python3 python3-pip python3-dev

# Install GPU mining dependencies
pip3 install cupy-cuda12x pycuda numpy requests
```

### 5. **Download the GPU miner**
```bash
# Create mining directory
mkdir -p /opt/bckn-mining
cd /opt/bckn-mining

# Download the miner
wget https://raw.githubusercontent.com/httptim/Bckn/master/bckn-miner-gpu.py
chmod +x bckn-miner-gpu.py
```

### 6. **Generate a mining address** (if you don't have one)
```bash
# Download address generator
wget https://raw.githubusercontent.com/httptim/Bckn/master/generate-address.py
chmod +x generate-address.py

# Generate address
python3 generate-address.py
```
Save the private key securely!

### 7. **Start mining**
```bash
# Mine with your private key
python3 bckn-miner-gpu.py YOUR_PRIVATE_KEY

# Or mine to a specific node
python3 bckn-miner-gpu.py YOUR_PRIVATE_KEY https://bckn.dev
```

## 📊 Expected Performance

With your H100 GPU:
- **Expected hashrate**: 500-1000 MH/s
- **Power consumption**: ~350W
- **Blocks per day**: Depends on network difficulty

## 🔧 Advanced Setup

### Run miner in background with systemd
```bash
# Create systemd service
cat > /etc/systemd/system/bckn-miner.service << EOF
[Unit]
Description=Bckn GPU Miner
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/bckn-mining
ExecStart=/usr/bin/python3 /opt/bckn-mining/bckn-miner-gpu.py YOUR_PRIVATE_KEY
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable bckn-miner
systemctl start bckn-miner

# Check status
systemctl status bckn-miner

# View logs
journalctl -u bckn-miner -f
```

### Monitor GPU usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvidia-smi dmon for detailed stats
nvidia-smi dmon
```

### Optimize performance
```bash
# Set GPU to persistence mode
nvidia-smi -pm 1

# Set maximum performance mode
nvidia-smi -pl 350  # Set power limit to 350W

# Lock GPU clocks for consistent performance (optional)
nvidia-smi -lgc 1980  # Lock at 1980 MHz
```

## 🛡️ Security Tips

1. **Use a dedicated mining address** - Don't use your main wallet
2. **Secure your private key** - Store it encrypted
3. **Monitor your droplet** - Set up alerts for high CPU/GPU usage
4. **Use firewall** - Only allow SSH and outbound HTTPS

```bash
# Basic firewall setup
ufw allow 22/tcp
ufw allow out 443/tcp
ufw --force enable
```

## 📈 Monitoring Dashboard

Create a simple monitoring script:
```bash
cat > /opt/bckn-mining/monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== Bckn Mining Monitor ==="
    echo "Date: $(date)"
    echo ""
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits
    echo ""
    echo "=== Miner Status ==="
    systemctl status bckn-miner --no-pager | grep -E "Active:|Main PID:"
    echo ""
    echo "=== Recent Logs ==="
    journalctl -u bckn-miner -n 5 --no-pager
    sleep 5
done
EOF

chmod +x /opt/bckn-mining/monitor.sh
```

Run with: `/opt/bckn-mining/monitor.sh`

## 🆘 Troubleshooting

### CUDA not found
```bash
# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Miner crashes
```bash
# Check for errors
journalctl -u bckn-miner -n 100

# Test GPU
python3 -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### Low hashrate
- Check GPU temperature: `nvidia-smi -q -d TEMPERATURE`
- Ensure GPU is not throttling
- Try increasing batch size in the miner

## 💰 Profit Estimation

With H100 at 750 MH/s:
- Network difficulty adjusts automatically
- Each block = 25 BCN
- Mining is competitive - join early for best results!

Happy mining! 🥓⛏️