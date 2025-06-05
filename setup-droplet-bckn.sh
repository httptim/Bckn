#!/bin/bash

# Bckn (Bckn) Server Setup Script for Digital Ocean Droplet
# Builds from YOUR fork with mining enabled

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOMAIN="bckn.dev"
WS_DOMAIN="ws.bckn.dev"
GITHUB_REPO="https://github.com/httptim/Bckn"
LETS_ENCRYPT_DIR="/var/www/letsencrypt"

echo -e "${BLUE}=== Bckn (Bckn Fork) Setup Script ===${NC}"
echo -e "${BLUE}Building from: ${GITHUB_REPO}${NC}"
echo -e "${BLUE}Domain: ${DOMAIN}${NC}"
echo -e "${BLUE}WebSocket Domain: ${WS_DOMAIN}${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Generate secure passwords
echo -e "${YELLOW}Generating secure passwords...${NC}"
DB_ROOT_PASS=$(openssl rand -hex 32)
DB_BCKN_PASS=$(openssl rand -hex 32)
REDIS_PASS=$(openssl rand -hex 32)
PROMETHEUS_PASS=$(openssl rand -hex 16)

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
apt-get update -qq
apt-get upgrade -y -qq

# Install dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt-get install -y -qq \
    curl \
    git \
    build-essential \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw \
    fail2ban \
    htop \
    wget \
    gnupg \
    lsb-release \
    ca-certificates

# Install Docker
echo -e "${YELLOW}Installing Docker...${NC}"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl start docker
systemctl enable docker

# Install MariaDB
echo -e "${YELLOW}Installing MariaDB...${NC}"
apt-get install -y -qq mariadb-server mariadb-client
systemctl start mariadb
systemctl enable mariadb

# Wait for MariaDB
sleep 5

# Setup MariaDB
echo -e "${YELLOW}Configuring MariaDB...${NC}"
mysql <<EOF || true
ALTER USER 'root'@'localhost' IDENTIFIED BY '${DB_ROOT_PASS}';
DELETE FROM mysql.global_priv WHERE User='';
DELETE FROM mysql.global_priv WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');
DROP DATABASE IF EXISTS test;
DELETE FROM mysql.db WHERE Db='test' OR Db='test\\_%';
FLUSH PRIVILEGES;
EOF

# Create Bckn database
mysql -u root -p"${DB_ROOT_PASS}" <<EOF
CREATE DATABASE IF NOT EXISTS bckn CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'bckn'@'localhost' IDENTIFIED BY '${DB_BCKN_PASS}';
CREATE USER IF NOT EXISTS 'bckn'@'%' IDENTIFIED BY '${DB_BCKN_PASS}';
GRANT ALL PRIVILEGES ON bckn.* TO 'bckn'@'localhost';
GRANT ALL PRIVILEGES ON bckn.* TO 'bckn'@'%';
FLUSH PRIVILEGES;
EOF

# Configure MariaDB to listen on all interfaces
sed -i 's/^bind-address.*/bind-address = 0.0.0.0/' /etc/mysql/mariadb.conf.d/50-server.cnf
systemctl restart mariadb

# Install Redis
echo -e "${YELLOW}Installing Redis...${NC}"
apt-get install -y -qq redis-server

# Configure Redis
echo -e "${YELLOW}Configuring Redis...${NC}"
cat > /etc/redis/redis.conf <<EOF
# Redis Configuration
bind 0.0.0.0 ::
protected-mode yes
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300
daemonize no
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
always-show-logo no
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis
requirepass ${REDIS_PASS}
maxmemory-policy allkeys-lru
EOF

chown redis:redis /etc/redis/redis.conf
chmod 640 /etc/redis/redis.conf
systemctl restart redis-server
systemctl enable redis-server

# Clone and build Bckn
echo -e "${YELLOW}Cloning and building Bckn...${NC}"
cd /opt
git clone ${GITHUB_REPO} bckn-source
cd bckn-source

# Build Docker image
docker build -t bckn:latest .

# Create docker-compose
mkdir -p /opt/bckn
cat > /opt/bckn/docker-compose.yml <<EOF
version: "3.9"
services:
  bckn:
    image: "bckn:latest"
    container_name: bckn
    network_mode: "host"
    environment:
      - DB_HOST=localhost
      - DB_USER=bckn
      - DB_PASS=${DB_BCKN_PASS}
      - DB_NAME=bckn
      - REDIS_HOST=localhost
      - REDIS_PASS=${REDIS_PASS}
      - PUBLIC_URL=${DOMAIN}
      - PUBLIC_WS_URL=${WS_DOMAIN}
      - NODE_ENV=production
      - USE_PROMETHEUS=true
      - PROMETHEUS_PASSWORD=${PROMETHEUS_PASS}
      - TRUST_PROXY_COUNT=1
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOF

# Configure Nginx (HTTP only first)
echo -e "${YELLOW}Configuring Nginx...${NC}"
mkdir -p ${LETS_ENCRYPT_DIR}
rm -f /etc/nginx/sites-enabled/default

cat > /etc/nginx/sites-available/${DOMAIN} <<EOF
server {
    listen 80;
    server_name ${DOMAIN};
    
    location /.well-known/acme-challenge/ {
        root ${LETS_ENCRYPT_DIR};
    }
    
    location / {
        return 200 'Setting up SSL...';
        add_header Content-Type text/plain;
    }
}
EOF

cat > /etc/nginx/sites-available/${WS_DOMAIN} <<EOF
server {
    listen 80;
    server_name ${WS_DOMAIN};
    
    location /.well-known/acme-challenge/ {
        root ${LETS_ENCRYPT_DIR};
    }
    
    location / {
        return 200 '{"ok":true,"message":"Setting up SSL..."}';
        add_header Content-Type application/json;
    }
}
EOF

ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/${WS_DOMAIN} /etc/nginx/sites-enabled/
nginx -t
systemctl enable nginx
systemctl restart nginx

# Get SSL certificates
echo -e "${YELLOW}Obtaining SSL certificates...${NC}"
certbot certonly \
    --webroot \
    -w ${LETS_ENCRYPT_DIR} \
    -d ${DOMAIN} \
    -d ${WS_DOMAIN} \
    --non-interactive \
    --agree-tos \
    --email admin@${DOMAIN} \
    --no-eff-email

# Configure Nginx with HTTPS
echo -e "${YELLOW}Configuring Nginx with HTTPS...${NC}"
cat > /etc/nginx/sites-available/${DOMAIN} <<'EOF'
server {
    listen 80;
    server_name bckn.dev;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    http2 on;
    server_name bckn.dev;

    ssl_certificate /etc/letsencrypt/live/bckn.dev/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bckn.dev/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers off;

    access_log /var/log/nginx/bckn.dev.access.log;
    error_log /var/log/nginx/bckn.dev.error.log;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,privatekey,Idempotency-Key' always;
        
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,privatekey,Idempotency-Key';
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }
    }

    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json application/xml+rss;
}
EOF

cat > /etc/nginx/sites-available/${WS_DOMAIN} <<'EOF'
server {
    listen 80;
    server_name ws.bckn.dev;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    http2 on;
    server_name ws.bckn.dev;

    ssl_certificate /etc/letsencrypt/live/bckn.dev/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bckn.dev/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    access_log /var/log/nginx/ws.bckn.dev.access.log;
    error_log /var/log/nginx/ws.bckn.dev.error.log;

    location /ws/gateway {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_buffering off;
        
        add_header 'Access-Control-Allow-Origin' '*' always;
    }

    location / {
        return 200 '{"ok":true,"ws_url":"wss://ws.bckn.dev/ws/gateway"}';
        add_header Content-Type application/json;
        add_header 'Access-Control-Allow-Origin' '*' always;
    }
}
EOF

nginx -t && systemctl reload nginx

# Configure firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw --force reset >/dev/null 2>&1
ufw default deny incoming >/dev/null 2>&1
ufw default allow outgoing >/dev/null 2>&1
ufw allow 22/tcp >/dev/null 2>&1
ufw allow 80/tcp >/dev/null 2>&1
ufw allow 443/tcp >/dev/null 2>&1
echo "y" | ufw enable >/dev/null 2>&1

# Start Bckn
echo -e "${YELLOW}Starting Bckn...${NC}"
cd /opt/bckn
docker compose up -d

# Wait for startup
sleep 10

# Enable mining in Redis
echo -e "${YELLOW}Enabling mining...${NC}"
redis-cli -a "${REDIS_PASS}" SET bckn:mining-enabled "true"
redis-cli -a "${REDIS_PASS}" SET bckn:transactions-enabled "true"

# Create genesis block
echo -e "${YELLOW}Creating genesis block...${NC}"
mysql -u bckn -p"${DB_BCKN_PASS}" bckn <<EOF
INSERT INTO blocks (
    id,
    address,
    value,
    hash,
    nonce,
    difficulty,
    time
) VALUES (
    1,
    'b00000000a',
    50,
    '0000000000000000000000000000000000000000000000000000000000000000',
    '',
    100000,
    NOW()
) ON DUPLICATE KEY UPDATE id=id;
EOF

# Restart Bckn to recognize genesis block
docker restart bckn
sleep 5

# Create update script
cat > /opt/bckn/update.sh <<'EOF'
#!/bin/bash
cd /opt/bckn-source
git pull
docker build -t bckn:latest .
cd /opt/bckn
docker compose down
docker compose up -d
echo "Update complete!"
EOF
chmod +x /opt/bckn/update.sh

# Save credentials
cat > /root/bckn-credentials.txt <<EOF
=== Bckn (Bckn Fork) Server Credentials ===
Generated on: $(date)

Database:
  Host: localhost
  Database: bckn
  User: bckn
  Password: ${DB_BCKN_PASS}
  Root Password: ${DB_ROOT_PASS}

Redis:
  Host: localhost
  Port: 6379
  Password: ${REDIS_PASS}

Prometheus:
  URL: https://${DOMAIN}/prometheus
  User: prometheus
  Password: ${PROMETHEUS_PASS}

URLs:
  Main: https://${DOMAIN}
  WebSocket: https://${WS_DOMAIN}/ws/gateway

Commands:
  View logs: docker logs -f bckn
  Restart: docker restart bckn
  Update from GitHub: /opt/bckn/update.sh
  Check mining status: redis-cli -a ${REDIS_PASS} GET bckn:mining-enabled
EOF
chmod 600 /root/bckn-credentials.txt

# Final output
echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${YELLOW}Testing endpoints...${NC}"
if curl -s https://${DOMAIN}/motd | grep -q "motd"; then
    echo -e "${GREEN}✓ API is working${NC}"
else
    echo -e "${YELLOW}API still starting up...${NC}"
fi
echo ""
echo -e "${RED}Credentials saved to: /root/bckn-credentials.txt${NC}"
echo -e "${GREEN}Your Bckn server is running at https://${DOMAIN}${NC}"
echo -e "${GREEN}Mining is ENABLED!${NC}"
echo ""
echo "To update from your GitHub repo later: /opt/bckn/update.sh"