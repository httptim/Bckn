#!/bin/bash
# Bckn Miner Loop Script - Runs continuously until stopped

if [ $# -eq 0 ]; then
    echo "Usage: $0 <private_key>"
    echo "Example: $0 your_private_key_here"
    exit 1
fi

PRIVATE_KEY=$1
MINER_PATH="/usr/local/bin/bckn-miner-cpu"
BLOCKS_FOUND=0
START_TIME=$(date +%s)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== Bckn Continuous Miner ===${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Trap Ctrl+C to show stats before exiting
trap 'show_stats' INT

show_stats() {
    echo ""
    echo -e "${CYAN}=== Mining Session Summary ===${NC}"
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo -e "Total blocks found: ${GREEN}$BLOCKS_FOUND${NC}"
    echo -e "Total BCN earned: ${GREEN}$((BLOCKS_FOUND * 25))${NC}"
    echo -e "Session duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    exit 0
}

# Main mining loop
while true; do
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] Starting new mining round...${NC}"
    
    # Run the miner and capture output
    OUTPUT=$($MINER_PATH "$PRIVATE_KEY" 2>&1)
    EXIT_CODE=$?
    
    # Check if block was found
    if echo "$OUTPUT" | grep -q "Block submitted successfully"; then
        BLOCKS_FOUND=$((BLOCKS_FOUND + 1))
        echo -e "${GREEN}✓ Block #$BLOCKS_FOUND found! Total earned: $((BLOCKS_FOUND * 25)) BCN${NC}"
        
        # Optional: Play a sound notification (if available)
        if command -v aplay &> /dev/null; then
            # Try to play system beep
            echo -e '\a'
        fi
    else
        echo -e "${YELLOW}No block found in this round${NC}"
    fi
    
    # Show the miner output
    echo "$OUTPUT" | tail -n 20
    
    # Small delay before restarting
    echo -e "${CYAN}Restarting in 3 seconds...${NC}"
    sleep 3
    echo ""
done