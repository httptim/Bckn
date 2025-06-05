#!/bin/bash

# Krist Server Docker Setup Script for Digital Ocean Droplet
# This script sets up Krist using Docker as recommended in the README

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="bckn.dev"
WS_DOMAIN="ws.bckn.dev"
DOCKER_GATEWAY="172.17.0.1"  # Default Docker gateway

echo -e "${BLUE}=== Krist Server Docker Setup Script ===${NC}"
echo -e "${BLUE}Domain: ${DOMAIN}${NC}"
echo -e "${BLUE}WebSocket Domain: ${WS_DOMAIN}${NC}"
echo -e "${BLUE}Ubuntu Version: $(lsb_release -d | cut -f2)${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Function to check if a command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 completed successfully${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Generate secure passwords
echo -e "${YELLOW}Generating secure passwords...${NC}"
DB_ROOT_PASS=$(openssl rand -hex 32)
DB_KRIST_PASS=$(openssl rand -hex 32)
REDIS_PASS=$(openssl rand -hex 32)
PROMETHEUS_PASS=$(openssl rand -hex 16)

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
apt-get update -qq
apt-get upgrade -y -qq
check_status "System update"

# Install dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt-get install -y -qq \
    curl \
    git \
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
check_status "System dependencies installation"

# Install Docker
echo -e "${YELLOW}Installing Docker...${NC}"
# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up the repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
check_status "Docker installation"

# Start and enable Docker
systemctl start docker
systemctl enable docker
check_status "Docker service startup"

# Install MariaDB
echo -e "${YELLOW}Installing MariaDB...${NC}"
apt-get install -y -qq mariadb-server mariadb-client
check_status "MariaDB installation"

# Start and enable MariaDB
systemctl start mariadb
systemctl enable mariadb
check_status "MariaDB service startup"

# Wait for MariaDB to be ready
echo -e "${YELLOW}Waiting for MariaDB to be ready...${NC}"
for i in {1..30}; do
    if mysqladmin ping &>/dev/null; then
        echo -e "${GREEN}MariaDB is ready${NC}"
        break
    fi
    sleep 1
done

# Setup MariaDB
echo -e "${YELLOW}Configuring MariaDB...${NC}"
mysql <<EOF || true
-- Set root password and secure installation
ALTER USER 'root'@'localhost' IDENTIFIED BY '${DB_ROOT_PASS}';
DELETE FROM mysql.global_priv WHERE User='';
DELETE FROM mysql.global_priv WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');
DROP DATABASE IF EXISTS test;
DELETE FROM mysql.db WHERE Db='test' OR Db='test\\_%';
FLUSH PRIVILEGES;
EOF

# Create Krist database and user (allow connections from Docker)
echo -e "${YELLOW}Creating Krist database...${NC}"
mysql -u root -p"${DB_ROOT_PASS}" <<EOF
CREATE DATABASE IF NOT EXISTS krist CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'krist'@'localhost' IDENTIFIED BY '${DB_KRIST_PASS}';
CREATE USER IF NOT EXISTS 'krist'@'${DOCKER_GATEWAY}' IDENTIFIED BY '${DB_KRIST_PASS}';
CREATE USER IF NOT EXISTS 'krist'@'172.%.%.%' IDENTIFIED BY '${DB_KRIST_PASS}';
GRANT ALL PRIVILEGES ON krist.* TO 'krist'@'localhost';
GRANT ALL PRIVILEGES ON krist.* TO 'krist'@'${DOCKER_GATEWAY}';
GRANT ALL PRIVILEGES ON krist.* TO 'krist'@'172.%.%.%';
FLUSH PRIVILEGES;
EOF
check_status "Krist database creation"

# Configure MariaDB to listen on Docker interface
echo -e "${YELLOW}Configuring MariaDB for Docker access...${NC}"
sed -i 's/^bind-address.*/bind-address = 0.0.0.0/' /etc/mysql/mariadb.conf.d/50-server.cnf
systemctl restart mariadb
check_status "MariaDB Docker configuration"

# Install Redis
echo -e "${YELLOW}Installing Redis...${NC}"
apt-get install -y -qq redis-server
check_status "Redis installation"

# Configure Redis for Docker access
echo -e "${YELLOW}Configuring Redis...${NC}"
cat > /etc/redis/redis.conf <<EOF
# Redis Configuration for Krist
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
check_status "Redis configuration"

# Test Redis connection
echo -e "${YELLOW}Testing Redis connection...${NC}"
if redis-cli -a "${REDIS_PASS}" ping | grep -q PONG; then
    echo -e "${GREEN}Redis is working correctly${NC}"
else
    echo -e "${RED}Redis connection failed${NC}"
    exit 1
fi

# Create directory for Let's Encrypt
echo -e "${YELLOW}Creating Let's Encrypt directory...${NC}"
mkdir -p /var/www/letsencrypt
check_status "Let's Encrypt directory creation"

# Configure Nginx - Step 1: HTTP only for SSL certificates
echo -e "${YELLOW}Configuring Nginx for SSL certificate generation...${NC}"

# Remove default site
rm -f /etc/nginx/sites-enabled/default

# Create HTTP configurations
cat > /etc/nginx/sites-available/${DOMAIN} <<EOF
server {
    listen 80;
    server_name ${DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }

    location / {
        return 200 'Krist server is being configured...';
        add_header Content-Type text/plain;
    }
}
EOF

cat > /etc/nginx/sites-available/${WS_DOMAIN} <<EOF
server {
    listen 80;
    server_name ${WS_DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }

    location / {
        return 200 '{"ok":true,"message":"SSL certificate pending..."}';
        add_header Content-Type application/json;
    }
}
EOF

# Enable sites
ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/${WS_DOMAIN} /etc/nginx/sites-enabled/

# Test and start Nginx
nginx -t
check_status "Nginx configuration test"

systemctl enable nginx
systemctl restart nginx
check_status "Nginx startup"

# Get SSL certificates
echo -e "${YELLOW}Obtaining SSL certificates...${NC}"
certbot certonly \
    --webroot \
    -w /var/www/letsencrypt \
    -d ${DOMAIN} \
    -d ${WS_DOMAIN} \
    --non-interactive \
    --agree-tos \
    --email admin@${DOMAIN} \
    --no-eff-email
check_status "SSL certificate generation"

# Configure Nginx - Step 2: Full HTTPS configuration
echo -e "${YELLOW}Configuring Nginx with HTTPS...${NC}"

# Main domain - Full HTTPS
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

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/bckn.dev/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bckn.dev/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Logging
    access_log /var/log/nginx/bckn.dev.access.log;
    error_log /var/log/nginx/bckn.dev.error.log;

    # Proxy to Krist Docker container
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
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization,privatekey,Idempotency-Key' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range,X-Request-ID' always;
        
        # Handle OPTIONS
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

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json application/xml+rss;
}
EOF

# WebSocket domain - Full HTTPS
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

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/bckn.dev/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bckn.dev/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers off;

    # Logging
    access_log /var/log/nginx/ws.bckn.dev.access.log;
    error_log /var/log/nginx/ws.bckn.dev.error.log;

    # WebSocket proxy
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
        
        # CORS
        add_header 'Access-Control-Allow-Origin' '*' always;
    }

    location / {
        return 200 '{"ok":true,"ws_url":"wss://ws.bckn.dev/ws/gateway"}';
        add_header Content-Type application/json;
        add_header 'Access-Control-Allow-Origin' '*' always;
    }
}
EOF

# Reload Nginx
nginx -t
systemctl reload nginx
check_status "Nginx HTTPS configuration"

# Create Docker Compose file
echo -e "${YELLOW}Creating Docker Compose configuration...${NC}"
mkdir -p /opt/krist
cat > /opt/krist/docker-compose.yml <<EOF
version: "3.9"
services:
  krist:
    image: "ghcr.io/tmpim/krist:latest"
    container_name: krist
    environment:
      - DB_HOST=${DOCKER_GATEWAY}
      - DB_USER=krist
      - DB_PASS=${DB_KRIST_PASS}
      - DB_NAME=krist
      - REDIS_HOST=${DOCKER_GATEWAY}
      - REDIS_PASS=${REDIS_PASS}
      - PUBLIC_URL=${DOMAIN}
      - PUBLIC_WS_URL=${WS_DOMAIN}
      - NODE_ENV=production
      - USE_PROMETHEUS=true
      - PROMETHEUS_PASSWORD=${PROMETHEUS_PASS}
      - TRUST_PROXY_COUNT=1
    ports:
      - "127.0.0.1:8080:8080"
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOF

# Pull and start Krist
echo -e "${YELLOW}Starting Krist with Docker...${NC}"
cd /opt/krist
docker compose pull
docker compose up -d
check_status "Krist Docker startup"

# Wait for Krist to start
echo -e "${YELLOW}Waiting for Krist to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8080 >/dev/null 2>&1; then
        echo -e "${GREEN}Krist is responding${NC}"
        break
    fi
    sleep 1
done

# Configure UFW firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw --force reset >/dev/null 2>&1
ufw default deny incoming >/dev/null 2>&1
ufw default allow outgoing >/dev/null 2>&1
ufw allow 22/tcp >/dev/null 2>&1
ufw allow 80/tcp >/dev/null 2>&1
ufw allow 443/tcp >/dev/null 2>&1
echo "y" | ufw enable >/dev/null 2>&1
check_status "Firewall configuration"

# Configure fail2ban
echo -e "${YELLOW}Configuring fail2ban...${NC}"
systemctl enable fail2ban >/dev/null 2>&1
systemctl start fail2ban >/dev/null 2>&1
check_status "Fail2ban configuration"

# Create backup script
echo -e "${YELLOW}Creating backup script...${NC}"
cat > /opt/krist/backup.sh <<EOF
#!/bin/bash
BACKUP_DIR="/opt/krist/backups"
DATE=\$(date +%Y%m%d_%H%M%S)
mkdir -p \$BACKUP_DIR

# Backup database
mysqldump -u krist -p${DB_KRIST_PASS} krist | gzip > \$BACKUP_DIR/krist_db_\$DATE.sql.gz

# Keep only last 7 days
find \$BACKUP_DIR -name "krist_db_*.sql.gz" -mtime +7 -delete
EOF
chmod +x /opt/krist/backup.sh

# Setup cron jobs
echo -e "${YELLOW}Setting up automatic tasks...${NC}"
(crontab -l 2>/dev/null || true; echo "0 3 * * * /opt/krist/backup.sh") | crontab -
(crontab -l 2>/dev/null || true; echo "0 0 * * 0 certbot renew --quiet --post-hook 'systemctl reload nginx'") | crontab -
(crontab -l 2>/dev/null || true; echo "0 4 * * * docker system prune -af --volumes") | crontab -
check_status "Cron jobs setup"

# Save credentials
echo -e "${YELLOW}Saving credentials...${NC}"
cat > /root/krist-credentials.txt <<EOF
=== Krist Server Credentials ===
Generated on: $(date)

Database:
  Host: ${DOCKER_GATEWAY} (from Docker container)
  Database: krist
  User: krist
  Password: ${DB_KRIST_PASS}
  Root Password: ${DB_ROOT_PASS}

Redis:
  Host: ${DOCKER_GATEWAY} (from Docker container)
  Port: 6379
  Password: ${REDIS_PASS}

Prometheus:
  URL: https://${DOMAIN}/prometheus
  User: prometheus
  Password: ${PROMETHEUS_PASS}

URLs:
  Main: https://${DOMAIN}
  WebSocket: https://${WS_DOMAIN}/ws/gateway
  API Docs: https://${DOMAIN}/docs

Docker Commands:
  View logs: docker logs -f krist
  Restart: docker restart krist
  Stop: docker stop krist
  Start: docker start krist
  Update: cd /opt/krist && docker compose pull && docker compose up -d

System Commands:
  Edit compose: nano /opt/krist/docker-compose.yml
  Backup now: /opt/krist/backup.sh
  View all logs: journalctl -f
EOF
chmod 600 /root/krist-credentials.txt

# Create convenience script
cat > /usr/local/bin/krist <<'EOF'
#!/bin/bash
case "$1" in
    logs)
        docker logs -f krist
        ;;
    restart)
        docker restart krist
        ;;
    stop)
        docker stop krist
        ;;
    start)
        docker start krist
        ;;
    status)
        docker ps | grep krist
        ;;
    update)
        cd /opt/krist
        docker compose pull
        docker compose up -d
        ;;
    *)
        echo "Usage: krist {logs|restart|stop|start|status|update}"
        exit 1
        ;;
esac
EOF
chmod +x /usr/local/bin/krist

# Final checks
echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${YELLOW}Checking services...${NC}"
echo -n "Docker: "
if docker ps | grep -q krist; then
    echo -e "${GREEN}✓ Krist container running${NC}"
else
    echo -e "${RED}✗ Krist container not running${NC}"
fi
echo -n "Main site: "
if curl -s -o /dev/null -w "%{http_code}" https://${DOMAIN} | grep -q "200\|301\|302"; then
    echo -e "${GREEN}✓ Working${NC}"
else
    echo -e "${YELLOW}Still initializing...${NC}"
fi
echo -n "WebSocket: "
if curl -s https://${WS_DOMAIN} | grep -q "ws_url"; then
    echo -e "${GREEN}✓ Working${NC}"
else
    echo -e "${YELLOW}Still initializing...${NC}"
fi
echo ""
echo -e "${RED}IMPORTANT: Credentials saved to /root/krist-credentials.txt${NC}"
echo -e "${GREEN}Your Krist server is now running at https://${DOMAIN}${NC}"
echo ""
echo -e "${YELLOW}Quick commands:${NC}"
echo "  krist logs     - View Krist logs"
echo "  krist restart  - Restart Krist"
echo "  krist status   - Check status"
echo "  krist update   - Update to latest version"
echo ""