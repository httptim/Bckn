#!/bin/bash
# Bckn CPU Miner Installation Script for RunPod Ubuntu
# Optimized for runpod/base:0.5.1-cpu

set -e

echo "==================================="
echo "Bckn CPU Miner Installation Script"
echo "==================================="
echo ""

# Update package lists
echo "Updating package lists..."
sudo apt-get update -qq

# Install required dependencies
echo "Installing dependencies..."
sudo apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    libcurl4-openssl-dev \
    libjsoncpp-dev \
    libssl-dev \
    git \
    wget \
    ca-certificates \
    pkg-config \
    cmake

# Check if jsoncpp headers are installed correctly
if [ ! -f "/usr/include/jsoncpp/json/json.h" ] && [ ! -f "/usr/include/json/json.h" ]; then
    echo "Installing jsoncpp from source..."
    cd /tmp
    git clone https://github.com/open-source-parsers/jsoncpp.git
    cd jsoncpp
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd /
    rm -rf /tmp/jsoncpp
fi

# Create build directory
BUILD_DIR="/tmp/bckn-miner-build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download the miner source files
echo ""
echo "Downloading miner source files..."
if [ -f "/workspace/bckn-miner-cpu.cpp" ]; then
    echo "Using local source files..."
    cp /workspace/bckn-miner-cpu.cpp .
    cp /workspace/Makefile .
else
    echo "Downloading from repository..."
    # Download from your GitHub repository
    wget -O bckn-miner-cpu.cpp https://raw.githubusercontent.com/httptim/Bckn/master/bckn-miner-cpu.cpp
    wget -O Makefile https://raw.githubusercontent.com/httptim/Bckn/master/Makefile
fi

# Compile the miner
echo ""
echo "Compiling miner (this may take a minute)..."
make clean
make -j$(nproc)

# Install the miner
echo ""
echo "Installing miner..."
sudo make install

# Create a simple run script
echo ""
echo "Creating run script..."
cat > /tmp/run-bckn-miner.sh << 'EOF'
#!/bin/bash
# Bckn Miner Runner Script

if [ $# -eq 0 ]; then
    echo "Usage: $0 <private_key>"
    echo "Example: $0 your_private_key_here"
    exit 1
fi

PRIVATE_KEY=$1

echo "Starting Bckn CPU Miner..."
echo "Press Ctrl+C to stop"
echo ""

# Run the miner
/usr/local/bin/bckn-miner-cpu "$PRIVATE_KEY"
EOF

chmod +x /tmp/run-bckn-miner.sh
sudo mv /tmp/run-bckn-miner.sh /usr/local/bin/run-bckn-miner

# Create systemd service (optional)
echo ""
echo "Creating systemd service..."
cat > /tmp/bckn-miner.service << 'EOF'
[Unit]
Description=Bckn CPU Miner
After=network.target

[Service]
Type=simple
User=nobody
Environment="PRIVATE_KEY=YOUR_PRIVATE_KEY_HERE"
ExecStart=/usr/local/bin/bckn-miner-cpu ${PRIVATE_KEY}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/bckn-miner.service /etc/systemd/system/

# Clean up
echo ""
echo "Cleaning up..."
cd /
rm -rf "$BUILD_DIR"

# Print instructions
echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo ""
echo "To run the miner manually:"
echo "  run-bckn-miner YOUR_PRIVATE_KEY"
echo ""
echo "To run as a service:"
echo "  1. Edit /etc/systemd/system/bckn-miner.service"
echo "  2. Replace YOUR_PRIVATE_KEY_HERE with your actual key"
echo "  3. Run: sudo systemctl enable bckn-miner"
echo "  4. Run: sudo systemctl start bckn-miner"
echo "  5. Check status: sudo systemctl status bckn-miner"
echo "  6. View logs: sudo journalctl -u bckn-miner -f"
echo ""
echo "The miner has been optimized for maximum CPU performance."
echo "Expected hashrate: 5-15 MH/s per core (depending on CPU)"
echo ""