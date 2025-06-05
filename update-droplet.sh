#!/bin/bash
# Script to update Bckn on droplet with your fork's code

echo "Updating Bckn from your Bckn fork..."

# SSH commands to run on the droplet
ssh root@$1 << 'EOF'
# Stop current container
cd /opt/bckn
docker compose down

# Clone/update your fork
if [ -d "/opt/bckn-source" ]; then
    cd /opt/bckn-source
    git pull
else
    git clone https://github.com/httptim/Bckn /opt/bckn-source
fi

# Build Docker image from your source
cd /opt/bckn-source
docker build -t bckn-custom .

# Update docker-compose to use custom image
cd /opt/bckn
sed -i 's|image: "ghcr.io/tmpim/bckn:latest"|image: "bckn-custom"|' docker-compose.yml

# Start with new image
docker compose up -d

# Enable mining in Redis
REDIS_PASS=$(grep REDIS_PASS docker-compose.yml | cut -d'=' -f2 | xargs)
redis-cli -a "$REDIS_PASS" SET bckn:mining-enabled "true"
redis-cli -a "$REDIS_PASS" SET bckn:transactions-enabled "true"

echo "Update complete! Mining enabled."
docker logs bckn --tail 20
EOF