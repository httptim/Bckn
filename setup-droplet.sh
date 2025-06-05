#!/bin/bash

# Krist Server Setup Script for Digital Ocean Droplet
# This script will set up everything needed to run Krist on bckn.dev

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="bckn.dev"
WS_DOMAIN="ws.bckn.dev"
KRIST_USER="krist"
KRIST_DIR="/home/krist/krist-server"
NODE_VERSION="20"

echo -e "${BLUE}=== Krist Server Setup Script ===${NC}"
echo -e "${BLUE}Domain: ${DOMAIN}${NC}"
echo -e "${BLUE}WebSocket Domain: ${WS_DOMAIN}${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Generate secure passwords (simplified to avoid special char issues)
echo -e "${YELLOW}Generating secure passwords...${NC}"
DB_ROOT_PASS=$(openssl rand -hex 32)
DB_KRIST_PASS=$(openssl rand -hex 32)
REDIS_PASS=$(openssl rand -hex 32)
PROMETHEUS_PASS=$(openssl rand -hex 16)

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
apt-get update
apt-get upgrade -y

# Install dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt-get install -y \
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
    software-properties-common

# Install Node.js
echo -e "${YELLOW}Installing Node.js v${NODE_VERSION}...${NC}"
curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
apt-get install -y nodejs

# Install pnpm
echo -e "${YELLOW}Installing pnpm...${NC}"
npm install -g pnpm

# Install MariaDB
echo -e "${YELLOW}Installing MariaDB...${NC}"
apt-get install -y mariadb-server mariadb-client

# Start MariaDB
systemctl start mariadb
systemctl enable mariadb

# Secure MariaDB installation
echo -e "${YELLOW}Securing MariaDB...${NC}"
# Use sudo mysql for initial setup (works with default Ubuntu MariaDB)
mysql <<EOF
-- Set root password
ALTER USER 'root'@'localhost' IDENTIFIED BY '${DB_ROOT_PASS}';

-- Remove anonymous users
DELETE FROM mysql.global_priv WHERE User='';

-- Remove remote root access
DELETE FROM mysql.global_priv WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');

-- Remove test database
DROP DATABASE IF EXISTS test;
DELETE FROM mysql.db WHERE Db='test' OR Db='test\\_%';

-- Create Krist database and user
CREATE DATABASE IF NOT EXISTS krist;
CREATE USER IF NOT EXISTS 'krist'@'localhost' IDENTIFIED BY '${DB_KRIST_PASS}';
GRANT ALL PRIVILEGES ON krist.* TO 'krist'@'localhost';

FLUSH PRIVILEGES;
EOF

# Install Redis
echo -e "${YELLOW}Installing Redis...${NC}"
apt-get install -y redis-server

# Configure Redis with password
echo -e "${YELLOW}Configuring Redis...${NC}"
# Use a more robust method to set Redis password
cat > /etc/redis/redis.conf.new << EOF
# Redis configuration
bind 127.0.0.1 ::1
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
always-show-logo yes
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

# Backup original and replace
mv /etc/redis/redis.conf /etc/redis/redis.conf.backup
mv /etc/redis/redis.conf.new /etc/redis/redis.conf
chown redis:redis /etc/redis/redis.conf
chmod 640 /etc/redis/redis.conf

systemctl restart redis-server
systemctl enable redis-server

# Create krist user
echo -e "${YELLOW}Creating krist user...${NC}"
if ! id -u ${KRIST_USER} >/dev/null 2>&1; then
    useradd -m -s /bin/bash ${KRIST_USER}
fi

# Clone Krist repository
echo -e "${YELLOW}Cloning Krist repository...${NC}"
if [ -d "${KRIST_DIR}" ]; then
    echo "Krist directory already exists, removing and re-cloning..."
    rm -rf ${KRIST_DIR}
fi
sudo -u ${KRIST_USER} git clone https://github.com/httptim/Bckn ${KRIST_DIR}

# Install dependencies and build
echo -e "${YELLOW}Installing dependencies and building...${NC}"
cd ${KRIST_DIR}
sudo -u ${KRIST_USER} pnpm install
sudo -u ${KRIST_USER} pnpm run build

# Create .env file
echo -e "${YELLOW}Creating .env file...${NC}"
cat > ${KRIST_DIR}/.env <<EOF
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=krist
DB_USER=krist
DB_PASS=${DB_KRIST_PASS}

# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASS=${REDIS_PASS}
REDIS_PREFIX=krist:

# Web Server Configuration
WEB_LISTEN=8080
PUBLIC_URL=https://${DOMAIN}
PUBLIC_WS_URL=https://${WS_DOMAIN}
TRUST_PROXY_COUNT=1

# Optional Features
NODE_ENV=production
USE_PROMETHEUS=true
PROMETHEUS_PASSWORD=${PROMETHEUS_PASS}

# Performance Settings
DB_POOL_MIN=5
DB_POOL_MAX=20
DB_POOL_IDLE_MS=300000
DB_POOL_ACQUIRE_MS=30000
DB_POOL_EVICT_MS=10000
EOF

chown ${KRIST_USER}:${KRIST_USER} ${KRIST_DIR}/.env
chmod 600 ${KRIST_DIR}/.env

# Create systemd service
echo -e "${YELLOW}Creating systemd service...${NC}"
cat > /etc/systemd/system/krist.service <<EOF
[Unit]
Description=Krist Server
After=network.target mysql.service redis.service
Requires=mysql.service redis.service

[Service]
Type=simple
User=${KRIST_USER}
WorkingDirectory=${KRIST_DIR}
ExecStart=/usr/bin/node ${KRIST_DIR}/dist/src/index.js
Restart=always
RestartSec=10

# Environment
Environment="NODE_ENV=production"

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${KRIST_DIR}

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=krist

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
echo -e "${YELLOW}Configuring Nginx...${NC}"

# Main domain configuration
cat > /etc/nginx/sites-available/${DOMAIN} <<'NGINX_CONFIG'
server {
    listen 80;
    server_name bckn.dev;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name bckn.dev;

    # SSL configuration will be added by certbot
    
    # Logging
    access_log /var/log/nginx/bckn.dev.access.log;
    error_log /var/log/nginx/bckn.dev.error.log;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Root directory for static files
    root /home/krist/krist-server/static;

    # Static files
    location ~ ^/(style\.css|favicon\.ico|logo2\.svg|down\.html|.*\.(png|jpg|jpeg|gif|ico))$ {
        try_files $uri =404;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }

    # API Documentation
    location /docs {
        alias /home/krist/krist-server/static/docs;
        try_files $uri $uri/ /docs/index.html;
    }

    # Error pages
    error_page 502 /down.html;
    location = /down.html {
        root /home/krist/krist-server/static;
        internal;
    }

    # Proxy to Krist
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
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range' always;
        
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS';
            add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization';
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json application/xml+rss;
}
NGINX_CONFIG

# WebSocket domain configuration
cat > /etc/nginx/sites-available/${WS_DOMAIN} <<'WS_CONFIG'
server {
    listen 80;
    server_name ws.bckn.dev;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ws.bckn.dev;

    # SSL configuration will be added by certbot
    
    # Logging
    access_log /var/log/nginx/ws.bckn.dev.access.log;
    error_log /var/log/nginx/ws.bckn.dev.error.log;

    # Error pages
    error_page 502 /down.html;
    location = /down.html {
        root /home/krist/krist-server/static;
        internal;
    }

    # WebSocket gateway
    location /ws/gateway {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific
        proxy_read_timeout 86400;
        proxy_buffering off;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
    }

    # Root endpoint
    location / {
        return 200 '{"ok":true,"ws_url":"wss://ws.bckn.dev/ws/gateway"}';
        add_header Content-Type application/json;
        add_header 'Access-Control-Allow-Origin' '*' always;
    }
}
WS_CONFIG

# Enable sites
ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/${WS_DOMAIN} /etc/nginx/sites-enabled/

# Remove default site
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t

# Configure UFW firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
echo "y" | ufw enable

# Configure fail2ban
echo -e "${YELLOW}Configuring fail2ban...${NC}"
systemctl enable fail2ban
systemctl start fail2ban

# Start services
echo -e "${YELLOW}Starting services...${NC}"
systemctl daemon-reload
systemctl enable nginx
systemctl restart nginx

# Get SSL certificates
echo -e "${YELLOW}Obtaining SSL certificates...${NC}"
certbot --nginx -d ${DOMAIN} -d ${WS_DOMAIN} --non-interactive --agree-tos --email admin@${DOMAIN} --redirect

# Generate API documentation
echo -e "${YELLOW}Generating API documentation...${NC}"
cd ${KRIST_DIR}
sudo -u ${KRIST_USER} pnpm run docs

# Start Krist service
echo -e "${YELLOW}Starting Krist service...${NC}"
systemctl enable krist
systemctl start krist

# Wait a moment for service to start
sleep 5

# Create backup script
echo -e "${YELLOW}Creating backup script...${NC}"
cat > /home/${KRIST_USER}/backup-krist.sh <<BACKUP_SCRIPT
#!/bin/bash
BACKUP_DIR="/home/krist/backups"
DATE=\$(date +%Y%m%d_%H%M%S)
mkdir -p \$BACKUP_DIR

# Backup database
mysqldump -u krist -p${DB_KRIST_PASS} krist | gzip > \$BACKUP_DIR/krist_db_\$DATE.sql.gz

# Keep only last 7 days of backups
find \$BACKUP_DIR -name "krist_db_*.sql.gz" -mtime +7 -delete
BACKUP_SCRIPT

chmod +x /home/${KRIST_USER}/backup-krist.sh
chown ${KRIST_USER}:${KRIST_USER} /home/${KRIST_USER}/backup-krist.sh

# Add cron job for backups
echo -e "${YELLOW}Setting up automatic backups...${NC}"
(crontab -u ${KRIST_USER} -l 2>/dev/null; echo "0 3 * * * /home/krist/backup-krist.sh") | crontab -u ${KRIST_USER} -

# Save credentials to file
echo -e "${YELLOW}Saving credentials...${NC}"
cat > /root/krist-credentials.txt <<EOF
=== Krist Server Credentials ===
Generated on: $(date)

Database Credentials:
MariaDB root password: ${DB_ROOT_PASS}
MariaDB krist password: ${DB_KRIST_PASS}

Redis Credentials:
Redis password: ${REDIS_PASS}

Prometheus Credentials:
Prometheus password: ${PROMETHEUS_PASS}

Access URLs:
Main site: https://${DOMAIN}
WebSocket: https://${WS_DOMAIN}/ws/gateway
API Docs: https://${DOMAIN}/docs
Prometheus metrics: https://${DOMAIN}/prometheus

Service Commands:
View logs: journalctl -u krist -f
Restart service: systemctl restart krist
Check status: systemctl status krist
Edit config: nano ${KRIST_DIR}/.env
EOF

chmod 600 /root/krist-credentials.txt

# Output credentials
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${RED}IMPORTANT: Save these credentials securely!${NC}"
echo ""
echo -e "${YELLOW}Database Credentials:${NC}"
echo "MariaDB root password: ${DB_ROOT_PASS}"
echo "MariaDB krist password: ${DB_KRIST_PASS}"
echo ""
echo -e "${YELLOW}Redis Credentials:${NC}"
echo "Redis password: ${REDIS_PASS}"
echo ""
echo -e "${YELLOW}Prometheus Credentials:${NC}"
echo "Prometheus password: ${PROMETHEUS_PASS}"
echo ""
echo -e "${YELLOW}Credentials saved to: /root/krist-credentials.txt${NC}"
echo ""
echo -e "${YELLOW}Service Status:${NC}"
systemctl status krist --no-pager || echo "Service may still be starting..."
echo ""
echo -e "${YELLOW}Access URLs:${NC}"
echo "Main site: https://${DOMAIN}"
echo "WebSocket: https://${WS_DOMAIN}/ws/gateway"
echo "API Docs: https://${DOMAIN}/docs"
echo "Prometheus metrics: https://${DOMAIN}/prometheus (password required)"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo "View logs: journalctl -u krist -f"
echo "Restart service: systemctl restart krist"
echo "Check status: systemctl status krist"
echo "Edit config: nano ${KRIST_DIR}/.env"
echo "View saved credentials: cat /root/krist-credentials.txt"
echo ""
echo -e "${GREEN}Setup complete! Your Krist server should now be running.${NC}"
echo -e "${GREEN}Check https://${DOMAIN} in a few moments.${NC}"