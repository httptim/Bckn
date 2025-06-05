#!/bin/bash
# Script to update Krist on droplet with your fork's code

echo "Updating Krist from your Bckn fork..."

# SSH commands to run on the droplet
ssh root@$1 << 'EOF'
# Stop current container
cd /opt/krist
docker compose down

# Clone/update your fork
if [ -d "/opt/krist-source" ]; then
    cd /opt/krist-source
    git pull
else
    git clone https://github.com/httptim/Bckn /opt/krist-source
fi

# Build Docker image from your source
cd /opt/krist-source
docker build -t krist-custom .

# Update docker-compose to use custom image
cd /opt/krist
sed -i 's|image: "ghcr.io/tmpim/krist:latest"|image: "krist-custom"|' docker-compose.yml

# Start with new image
docker compose up -d

# Enable mining in Redis
REDIS_PASS=$(grep REDIS_PASS docker-compose.yml | cut -d'=' -f2 | xargs)
redis-cli -a "$REDIS_PASS" SET krist:mining-enabled "true"
redis-cli -a "$REDIS_PASS" SET krist:transactions-enabled "true"

echo "Update complete! Mining enabled."
docker logs krist --tail 20
EOF