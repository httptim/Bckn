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

# Generate secure passwords
echo -e "${YELLOW}Generating secure passwords...${NC}"
DB_ROOT_PASS=$(openssl rand -base64 32)
DB_KRIST_PASS=$(openssl rand -base64 32)
REDIS_PASS=$(openssl rand -base64 32)
PROMETHEUS_PASS=$(openssl rand -base64 16)

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

# Secure MariaDB installation
echo -e "${YELLOW}Securing MariaDB...${NC}"
mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED BY '${DB_ROOT_PASS}'"
mysql -e "DELETE FROM mysql.global_priv WHERE User=''"
mysql -e "DELETE FROM mysql.global_priv WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1')"
mysql -e "DROP DATABASE IF EXISTS test"
mysql -e "DELETE FROM mysql.db WHERE Db='test' OR Db='test\\_%'"
mysql -e "FLUSH PRIVILEGES"

# Create Krist database and user
echo -e "${YELLOW}Creating Krist database and user...${NC}"
mysql -u root -p"${DB_ROOT_PASS}" <<EOF
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
sed -i "s/# requirepass foobared/requirepass ${REDIS_PASS}/" /etc/redis/redis.conf
sed -i "s/supervised no/supervised systemd/" /etc/redis/redis.conf
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
    echo "Krist directory already exists, pulling latest changes..."
    cd ${KRIST_DIR}
    sudo -u ${KRIST_USER} git pull
else
    sudo -u ${KRIST_USER} git clone https://github.com/httptim/Bckn ${KRIST_DIR}
fi

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
cat > /etc/nginx/sites-available/${DOMAIN} <<EOF
server {
    listen 80;
    server_name ${DOMAIN};
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ${DOMAIN};

    # SSL configuration will be added by certbot
    
    # Logging
    access_log /var/log/nginx/${DOMAIN}.access.log;
    error_log /var/log/nginx/${DOMAIN}.error.log;

    # Static files
    location /style.css {
        root ${KRIST_DIR}/static;
        try_files \$uri =404;
    }

    location /favicon.ico {
        root ${KRIST_DIR}/static;
        try_files \$uri =404;
    }

    location /logo2.svg {
        root ${KRIST_DIR}/static;
        try_files \$uri =404;
    }

    # Error pages
    error_page 502 /down.html;
    location = /down.html {
        root ${KRIST_DIR}/static;
        internal;
    }

    # Proxy to Krist
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range' always;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/json;
}
EOF

# WebSocket domain configuration
cat > /etc/nginx/sites-available/${WS_DOMAIN} <<EOF
server {
    listen 80;
    server_name ${WS_DOMAIN};
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ${WS_DOMAIN};

    # SSL configuration will be added by certbot
    
    # Logging
    access_log /var/log/nginx/${WS_DOMAIN}.access.log;
    error_log /var/log/nginx/${WS_DOMAIN}.error.log;

    # Error pages
    error_page 502 /down.html;
    location = /down.html {
        root ${KRIST_DIR}/static;
        internal;
    }

    # WebSocket gateway
    location /ws/gateway {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket specific
        proxy_read_timeout 86400;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range' always;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/json;
}
EOF

# Enable sites
ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/${WS_DOMAIN} /etc/nginx/sites-enabled/

# Remove default site
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t

# Configure UFW firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
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

# Start Krist service
echo -e "${YELLOW}Starting Krist service...${NC}"
systemctl enable krist
systemctl start krist

# Create backup script
echo -e "${YELLOW}Creating backup script...${NC}"
cat > /home/${KRIST_USER}/backup-krist.sh <<'EOF'
#!/bin/bash
BACKUP_DIR="/home/krist/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup database
mysqldump -u krist -p${DB_KRIST_PASS} krist | gzip > $BACKUP_DIR/krist_db_$DATE.sql.gz

# Keep only last 7 days of backups
find $BACKUP_DIR -name "krist_db_*.sql.gz" -mtime +7 -delete
EOF

chmod +x /home/${KRIST_USER}/backup-krist.sh
chown ${KRIST_USER}:${KRIST_USER} /home/${KRIST_USER}/backup-krist.sh

# Add cron job for backups
echo -e "${YELLOW}Setting up automatic backups...${NC}"
(crontab -u ${KRIST_USER} -l 2>/dev/null; echo "0 3 * * * /home/krist/backup-krist.sh") | crontab -u ${KRIST_USER} -

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
echo -e "${YELLOW}Service Status:${NC}"
systemctl status krist --no-pager
echo ""
echo -e "${YELLOW}Access URLs:${NC}"
echo "Main site: https://${DOMAIN}"
echo "WebSocket: https://${WS_DOMAIN}/ws/gateway"
echo "Prometheus metrics: https://${DOMAIN}/prometheus (password required)"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo "View logs: journalctl -u krist -f"
echo "Restart service: systemctl restart krist"
echo "Check status: systemctl status krist"
echo "Edit config: nano ${KRIST_DIR}/.env"
echo ""
echo -e "${GREEN}Setup complete! Your Krist server should now be running.${NC}"