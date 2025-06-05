#!/bin/bash

# Bckn Server Setup Script for Digital Ocean Droplet
# Tested on Ubuntu 24.10 with modern MariaDB, Redis, and Nginx

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
BCKN_USER="bckn"
BCKN_DIR="/home/bckn/bckn-server"
NODE_VERSION="20"
LETS_ENCRYPT_DIR="/var/www/letsencrypt"

echo -e "${BLUE}=== Bckn Server Setup Script ===${NC}"
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

# Generate secure passwords (using hex to avoid special character issues)
echo -e "${YELLOW}Generating secure passwords...${NC}"
DB_ROOT_PASS=$(openssl rand -hex 32)
DB_BCKN_PASS=$(openssl rand -hex 32)
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
    build-essential \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw \
    fail2ban \
    htop \
    wget \
    software-properties-common \
    gnupg \
    lsb-release
check_status "System dependencies installation"

# Install Node.js for Ubuntu 24.x
echo -e "${YELLOW}Installing Node.js v${NODE_VERSION}...${NC}"
mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_VERSION}.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
apt-get update -qq
apt-get install -y -qq nodejs
check_status "Node.js installation"

# Verify Node.js installation
node_version=$(node --version)
echo -e "${GREEN}Node.js installed: ${node_version}${NC}"

# Install pnpm
echo -e "${YELLOW}Installing pnpm...${NC}"
npm install -g pnpm@latest --silent
check_status "pnpm installation"

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

# Create Bckn database and user
echo -e "${YELLOW}Creating Bckn database...${NC}"
mysql -u root -p"${DB_ROOT_PASS}" <<EOF
CREATE DATABASE IF NOT EXISTS bckn CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'bckn'@'localhost' IDENTIFIED BY '${DB_BCKN_PASS}';
GRANT ALL PRIVILEGES ON bckn.* TO 'bckn'@'localhost';
FLUSH PRIVILEGES;
EOF
check_status "Bckn database creation"

# Install Redis
echo -e "${YELLOW}Installing Redis...${NC}"
apt-get install -y -qq redis-server
check_status "Redis installation"

# Configure Redis (write complete config to avoid sed issues)
echo -e "${YELLOW}Configuring Redis...${NC}"
cat > /etc/redis/redis.conf <<EOF
# Redis Configuration for Bckn
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
lazyfree-lazy-eviction no
lazyfree-lazy-expire no
lazyfree-lazy-server-del no
replica-lazy-flush no
appendonly no
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes
lua-time-limit 5000
slowlog-log-slower-than 10000
slowlog-max-len 128
latency-monitor-threshold 0
notify-keyspace-events ""
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100
activerehashing yes
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
hz 10
dynamic-hz yes
aof-rewrite-incremental-fsync yes
rdb-save-incremental-fsync yes
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

# Create bckn user
echo -e "${YELLOW}Creating bckn user...${NC}"
if ! id -u ${BCKN_USER} >/dev/null 2>&1; then
    useradd -m -s /bin/bash ${BCKN_USER}
    check_status "Bckn user creation"
else
    echo -e "${GREEN}Bckn user already exists${NC}"
fi

# Clone Bckn repository
echo -e "${YELLOW}Cloning Bckn repository...${NC}"
if [ -d "${BCKN_DIR}" ]; then
    rm -rf ${BCKN_DIR}
fi
sudo -u ${BCKN_USER} git clone https://github.com/httptim/Bckn ${BCKN_DIR}
check_status "Repository clone"

# Install dependencies and build
echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
cd ${BCKN_DIR}
sudo -u ${BCKN_USER} pnpm install --frozen-lockfile
check_status "Node.js dependencies installation"

echo -e "${YELLOW}Building Bckn...${NC}"
sudo -u ${BCKN_USER} pnpm run build
check_status "Bckn build"

# Create .env file
echo -e "${YELLOW}Creating environment configuration...${NC}"
cat > ${BCKN_DIR}/.env <<EOF
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=bckn
DB_USER=bckn
DB_PASS=${DB_BCKN_PASS}

# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASS=${REDIS_PASS}
REDIS_PREFIX=bckn:

# Web Server Configuration
WEB_LISTEN=8080
PUBLIC_URL=https://${DOMAIN}
PUBLIC_WS_URL=https://${WS_DOMAIN}
TRUST_PROXY_COUNT=1

# Environment
NODE_ENV=production
USE_PROMETHEUS=true
PROMETHEUS_PASSWORD=${PROMETHEUS_PASS}

# Database Pool Settings
DB_POOL_MIN=5
DB_POOL_MAX=20
DB_POOL_IDLE_MS=300000
DB_POOL_ACQUIRE_MS=30000
DB_POOL_EVICT_MS=10000
EOF

chown ${BCKN_USER}:${BCKN_USER} ${BCKN_DIR}/.env
chmod 600 ${BCKN_DIR}/.env
check_status "Environment configuration"

# Create systemd service
echo -e "${YELLOW}Creating systemd service...${NC}"
cat > /etc/systemd/system/bckn.service <<EOF
[Unit]
Description=Bckn Server
After=network.target mariadb.service redis-server.service
Requires=mariadb.service redis-server.service

[Service]
Type=simple
User=${BCKN_USER}
WorkingDirectory=${BCKN_DIR}
ExecStart=/usr/bin/node ${BCKN_DIR}/dist/src/index.js
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bckn

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${BCKN_DIR}

# Environment
Environment="NODE_ENV=production"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
check_status "Systemd service creation"

# Create Let's Encrypt webroot directory
echo -e "${YELLOW}Creating Let's Encrypt directory...${NC}"
mkdir -p ${LETS_ENCRYPT_DIR}
check_status "Let's Encrypt directory creation"

# Configure Nginx - Step 1: HTTP only for SSL certificates
echo -e "${YELLOW}Configuring Nginx for SSL certificate generation...${NC}"

# Remove default site if exists
rm -f /etc/nginx/sites-enabled/default

# Main domain - HTTP only
cat > /etc/nginx/sites-available/${DOMAIN} <<EOF
server {
    listen 80;
    server_name ${DOMAIN};

    # Let's Encrypt challenge location
    location /.well-known/acme-challenge/ {
        root ${LETS_ENCRYPT_DIR};
        try_files \$uri =404;
    }

    # Temporary response while setting up
    location / {
        return 200 'Bckn server is being configured...';
        add_header Content-Type text/plain;
    }
}
EOF

# WebSocket domain - HTTP only
cat > /etc/nginx/sites-available/${WS_DOMAIN} <<EOF
server {
    listen 80;
    server_name ${WS_DOMAIN};

    # Let's Encrypt challenge location
    location /.well-known/acme-challenge/ {
        root ${LETS_ENCRYPT_DIR};
        try_files \$uri =404;
    }

    # API response
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

# Wait for Nginx to be ready
sleep 2

# Get SSL certificates
echo -e "${YELLOW}Obtaining SSL certificates from Let's Encrypt...${NC}"
certbot certonly \
    --webroot \
    -w ${LETS_ENCRYPT_DIR} \
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
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Logging
    access_log /var/log/nginx/bckn.dev.access.log;
    error_log /var/log/nginx/bckn.dev.error.log;

    # Root directory
    root /home/bckn/bckn-server/static;

    # Static files with caching
    location ~ ^/(style\.css|favicon\.ico|logo2\.svg|down\.html|.*\.(png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot))$ {
        try_files $uri =404;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # API Documentation
    location /docs {
        alias /home/bckn/bckn-server/static/docs;
        try_files $uri $uri/ /docs/index.html;
    }

    # Error page
    error_page 502 503 504 /down.html;
    location = /down.html {
        internal;
    }

    # Prometheus metrics (protected)
    location /prometheus {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        auth_basic "Prometheus Metrics";
        auth_basic_user_file /etc/nginx/.htpasswd-prometheus;
    }

    # Main application proxy
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        
        # Headers
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

    # Compression
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
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Logging
    access_log /var/log/nginx/ws.bckn.dev.access.log;
    error_log /var/log/nginx/ws.bckn.dev.error.log;

    # Error page
    error_page 502 503 504 /down.html;
    location = /down.html {
        root /home/bckn/bckn-server/static;
        internal;
    }

    # WebSocket endpoint
    location /ws/gateway {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        
        # WebSocket headers
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific settings
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        proxy_buffering off;
        
        # CORS
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
    }

    # Root endpoint
    location / {
        default_type application/json;
        return 200 '{"ok":true,"ws_url":"wss://ws.bckn.dev/ws/gateway","motd":"Welcome to Bckn!"}';
        add_header 'Access-Control-Allow-Origin' '*' always;
    }
}
EOF

# Create htpasswd for Prometheus
echo -e "${YELLOW}Creating Prometheus authentication...${NC}"
echo -n "prometheus:$(openssl passwd -apr1 ${PROMETHEUS_PASS})" > /etc/nginx/.htpasswd-prometheus
check_status "Prometheus authentication setup"

# Test and reload Nginx
nginx -t
check_status "Nginx HTTPS configuration test"
systemctl reload nginx
check_status "Nginx reload with HTTPS"

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

# Generate API documentation
echo -e "${YELLOW}Generating API documentation...${NC}"
cd ${BCKN_DIR}
sudo -u ${BCKN_USER} pnpm run docs || echo -e "${YELLOW}API docs generation skipped (optional)${NC}"

# Start Bckn service
echo -e "${YELLOW}Starting Bckn service...${NC}"
systemctl enable bckn
systemctl start bckn
check_status "Bckn service startup"

# Wait for Bckn to start
echo -e "${YELLOW}Waiting for Bckn to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8080 >/dev/null 2>&1; then
        echo -e "${GREEN}Bckn is responding${NC}"
        break
    fi
    sleep 1
done

# Create backup script
echo -e "${YELLOW}Creating backup script...${NC}"
cat > /home/${BCKN_USER}/backup-bckn.sh <<'BACKUP'
#!/bin/bash
BACKUP_DIR="/home/bckn/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup database
DB_PASS=$(grep DB_PASS /home/bckn/bckn-server/.env | cut -d= -f2)
mysqldump -u bckn -p$DB_PASS bckn | gzip > $BACKUP_DIR/bckn_db_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "bckn_db_*.sql.gz" -mtime +7 -delete
BACKUP

chmod +x /home/${BCKN_USER}/backup-bckn.sh
chown ${BCKN_USER}:${BCKN_USER} /home/${BCKN_USER}/backup-bckn.sh

# Setup daily backup cron
echo -e "${YELLOW}Setting up automatic backups...${NC}"
(crontab -u ${BCKN_USER} -l 2>/dev/null || true; echo "0 3 * * * /home/bckn/backup-bckn.sh") | crontab -u ${BCKN_USER} -
check_status "Backup configuration"

# Setup auto-renewal for SSL certificates
echo -e "${YELLOW}Setting up SSL auto-renewal...${NC}"
(crontab -l 2>/dev/null || true; echo "0 0 * * 0 certbot renew --quiet --post-hook 'systemctl reload nginx'") | crontab -
check_status "SSL auto-renewal"

# Save credentials
echo -e "${YELLOW}Saving credentials...${NC}"
cat > /root/bckn-credentials.txt <<EOF
=== Bckn Server Credentials ===
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
  API Docs: https://${DOMAIN}/docs

Commands:
  View logs: journalctl -u bckn -f
  Restart: systemctl restart bckn
  Status: systemctl status bckn
  Edit config: nano ${BCKN_DIR}/.env
  Backup now: /home/bckn/backup-bckn.sh
EOF
chmod 600 /root/bckn-credentials.txt

# Final status check
echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${YELLOW}Checking service status...${NC}"
systemctl status bckn --no-pager | head -n 5 || true
echo ""
echo -e "${YELLOW}Testing endpoints...${NC}"
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
echo -e "${RED}IMPORTANT: Credentials saved to /root/bckn-credentials.txt${NC}"
echo -e "${GREEN}Your Bckn server is now running at https://${DOMAIN}${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Check the logs: journalctl -u bckn -f"
echo "2. Visit https://${DOMAIN} in your browser"
echo "3. Save the credentials from /root/bckn-credentials.txt"
echo ""