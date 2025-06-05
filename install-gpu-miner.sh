#!/bin/bash

# Bckn GPU Miner Installation Script for Digital Ocean GPU Droplet
# Supports Ubuntu 22.04 with NVIDIA H100 GPUs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root"
fi

# Header
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           Bckn GPU Miner Installer for Digital Ocean              ║"
echo "║                    NVIDIA H100 GPU Support                        ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get user input
read -p "Enter your Bckn private key (or press Enter to generate new): " PRIVATE_KEY
read -p "Enter Bckn node URL (default: https://bckn.dev): " NODE_URL
NODE_URL=${NODE_URL:-https://bckn.dev}

# System update
log "Updating system packages..."
apt update && apt upgrade -y

# Install basic dependencies
log "Installing basic dependencies..."
apt install -y \
    build-essential \
    curl \
    wget \
    git \
    htop \
    nvtop \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    software-properties-common \
    gnupg \
    lsb-release

# Check for existing NVIDIA drivers
if nvidia-smi &> /dev/null; then
    log "NVIDIA drivers already installed"
    nvidia-smi
else
    log "Installing NVIDIA drivers..."
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt update
    
    # Install NVIDIA driver and CUDA
    apt install -y nvidia-driver-535 cuda-12-3
    
    # Clean up
    rm cuda-keyring_1.1-1_all.deb
    
    warning "NVIDIA drivers installed. A reboot is required."
    warning "Please run this script again after rebooting."
    
    read -p "Reboot now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        reboot
    else
        error "Please reboot manually and run this script again."
    fi
fi

# Verify GPU is detected
log "Verifying GPU detection..."
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
log "Detected $GPU_COUNT GPU(s)"

# Set GPU persistence mode
log "Setting GPU persistence mode..."
nvidia-smi -pm 1

# Set maximum performance mode based on GPU type
log "Setting GPU performance mode..."

# Detect GPU model and set appropriate power limits
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log "Detected GPU: $GPU_NAME"

# Set power limits based on GPU model
if [[ $GPU_NAME == *"H100"* ]]; then
    POWER_LIMIT=350
elif [[ $GPU_NAME == *"RTX 5090"* ]] || [[ $GPU_NAME == *"RTX 4090"* ]]; then
    # RTX 5090/4090 typically support 450-600W
    POWER_LIMIT=500
elif [[ $GPU_NAME == *"RTX 4080"* ]]; then
    POWER_LIMIT=320
elif [[ $GPU_NAME == *"RTX 3090"* ]]; then
    POWER_LIMIT=350
elif [[ $GPU_NAME == *"A100"* ]]; then
    POWER_LIMIT=300
else
    # Skip power limit setting for unknown GPUs
    warning "Unknown GPU model. Skipping power limit configuration."
    POWER_LIMIT=0
fi

if [ $POWER_LIMIT -gt 0 ]; then
    log "Setting power limit to ${POWER_LIMIT}W..."
    nvidia-smi -pl $POWER_LIMIT || warning "Failed to set power limit. GPU may use default settings."
fi

# Install Python GPU libraries
log "Installing Python GPU libraries..."
pip3 install --upgrade pip

# Install CUDA Python packages
pip3 install \
    cupy-cuda12x \
    pycuda \
    numpy \
    requests \
    urllib3

# Create mining directory
log "Creating mining directory..."
MINING_DIR="/opt/bckn-mining"
mkdir -p $MINING_DIR
cd $MINING_DIR

# Download mining scripts
log "Downloading mining scripts..."
wget -O bckn-miner-gpu-enhanced.py https://raw.githubusercontent.com/httptim/Bckn/master/bckn-miner-gpu-enhanced.py
wget -O generate-address.py https://raw.githubusercontent.com/httptim/Bckn/master/generate-address.py
chmod +x bckn-miner-gpu-enhanced.py generate-address.py

# Generate address if not provided
if [ -z "$PRIVATE_KEY" ]; then
    log "Generating new Bckn address..."
    RESULT=$(python3 generate-address.py)
    echo -e "${GREEN}$RESULT${NC}"
    echo
    read -p "Enter the private key from above: " PRIVATE_KEY
fi

# Test the miner
log "Testing miner configuration..."
timeout 10 python3 bckn-miner-gpu-enhanced.py $PRIVATE_KEY --node $NODE_URL || true

# Create systemd service
log "Creating systemd service..."
cat > /etc/systemd/system/bckn-miner.service << EOF
[Unit]
Description=Bckn GPU Miner
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$MINING_DIR
ExecStart=/usr/bin/python3 $MINING_DIR/bckn-miner-gpu-enhanced.py $PRIVATE_KEY --background --node $NODE_URL
Restart=always
RestartSec=10
StandardOutput=append:/var/log/bckn-miner.log
StandardError=append:/var/log/bckn-miner.log

# GPU environment
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

[Install]
WantedBy=multi-user.target
EOF

# Create log rotation
log "Setting up log rotation..."
cat > /etc/logrotate.d/bckn-miner << EOF
/var/log/bckn-miner.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

# Create monitoring script
log "Creating monitoring script..."
cat > $MINING_DIR/monitor.sh << 'EOF'
#!/bin/bash
# Bckn Miner Monitor

while true; do
    clear
    echo "=== Bckn Mining Monitor ==="
    echo "Date: $(date)"
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,power.draw,memory.used --format=csv
    echo ""
    echo "=== Miner Status ==="
    systemctl status bckn-miner --no-pager | grep -E "Active:|Main PID:"
    echo ""
    echo "=== Recent Logs ==="
    tail -n 10 /var/log/bckn-miner.log
    echo ""
    echo "Press Ctrl+C to exit"
    sleep 5
done
EOF
chmod +x $MINING_DIR/monitor.sh

# Create helper scripts
log "Creating helper scripts..."

# Start script
cat > $MINING_DIR/start-mining.sh << EOF
#!/bin/bash
systemctl start bckn-miner
echo "Mining started. Check status with: systemctl status bckn-miner"
echo "View logs with: journalctl -u bckn-miner -f"
EOF
chmod +x $MINING_DIR/start-mining.sh

# Stop script
cat > $MINING_DIR/stop-mining.sh << EOF
#!/bin/bash
systemctl stop bckn-miner
echo "Mining stopped."
EOF
chmod +x $MINING_DIR/stop-mining.sh

# Interactive mode script
cat > $MINING_DIR/run-interactive.sh << EOF
#!/bin/bash
cd $MINING_DIR
python3 bckn-miner-gpu-enhanced.py $PRIVATE_KEY --node $NODE_URL
EOF
chmod +x $MINING_DIR/run-interactive.sh

# Setup firewall
log "Configuring firewall..."
ufw allow 22/tcp
ufw allow out 443/tcp
ufw --force enable || true

# Install monitoring tools
log "Installing additional monitoring tools..."
apt install -y nvtop htop iotop nethogs

# Create command aliases
log "Creating convenient aliases..."
cat >> /root/.bashrc << 'EOF'

# Bckn Mining Aliases
alias mining-start='systemctl start bckn-miner'
alias mining-stop='systemctl stop bckn-miner'
alias mining-status='systemctl status bckn-miner'
alias mining-logs='journalctl -u bckn-miner -f'
alias mining-monitor='/opt/bckn-mining/monitor.sh'
alias mining-gui='cd /opt/bckn-mining && python3 bckn-miner-gpu-enhanced.py'
alias gpu-watch='watch -n 1 nvidia-smi'
EOF

# Enable and start service
log "Enabling mining service..."
systemctl daemon-reload
systemctl enable bckn-miner

# Summary
echo
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Installation Complete!                         ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${BLUE}Mining Configuration:${NC}"
echo "  Private Key: ${PRIVATE_KEY:0:20}..."
echo "  Node URL: $NODE_URL"
echo "  GPU Count: $GPU_COUNT"
echo
echo -e "${BLUE}Quick Commands:${NC}"
echo "  Start Mining:        mining-start  (or systemctl start bckn-miner)"
echo "  Stop Mining:         mining-stop   (or systemctl stop bckn-miner)"
echo "  View Status:         mining-status (or systemctl status bckn-miner)"
echo "  Watch Logs:          mining-logs   (or journalctl -u bckn-miner -f)"
echo "  Monitor GPUs:        mining-monitor (or $MINING_DIR/monitor.sh)"
echo "  Run with GUI:        mining-gui"
echo "  Watch GPU Stats:     gpu-watch"
echo
echo -e "${BLUE}File Locations:${NC}"
echo "  Mining Directory:    $MINING_DIR"
echo "  Log File:           /var/log/bckn-miner.log"
echo "  Service File:       /etc/systemd/system/bckn-miner.service"
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Start mining with: ${GREEN}mining-start${NC}"
echo "2. Monitor performance with: ${GREEN}mining-monitor${NC}"
echo "3. Check earnings at: ${GREEN}$NODE_URL${NC}"
echo
echo -e "${GREEN}Happy Mining! 🥓⛏️${NC}"