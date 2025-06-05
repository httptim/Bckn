#!/bin/bash
# Bckn Admin Tool - Database Management Script
# Usage: ./bckn-admin.sh [command] [options]

# Database configuration - update these with your actual values
DB_HOST="localhost"
DB_USER="root"
DB_PASS="770f8169a320af3b91b56c1239d80fc57c01e716e0e8b690a13fc2a4443e877e"
DB_NAME="krist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# MySQL command
MYSQL="mysql -h $DB_HOST -u $DB_USER $DB_NAME"
if [ ! -z "$DB_PASS" ]; then
    MYSQL="mysql -h $DB_HOST -u $DB_USER -p$DB_PASS $DB_NAME"
fi

# Function to execute SQL query
exec_sql() {
    echo "$1" | $MYSQL 2>/dev/null
}

# Function to execute SQL query with results
query_sql() {
    echo "$1" | $MYSQL -t 2>/dev/null
}

# Display help
show_help() {
    echo -e "${CYAN}Bckn Admin Tool - Database Management${NC}"
    echo -e "${YELLOW}Usage:${NC} $0 [command] [options]"
    echo ""
    echo -e "${GREEN}Address Commands:${NC}"
    echo "  address-list [limit]              List all addresses (default: top 20 by balance)"
    echo "  address-show <address>            Show detailed info for an address"
    echo "  address-balance <address> <amt>   Set balance for an address"
    echo "  address-add <address> <balance>   Add new address with balance"
    echo "  address-lock <address>            Lock an address"
    echo "  address-unlock <address>          Unlock an address"
    echo "  address-alert <address> <msg>     Set alert message for address"
    echo ""
    echo -e "${GREEN}Name Commands:${NC}"
    echo "  name-list [owner]                 List all names or names by owner"
    echo "  name-show <name>                  Show detailed info for a name"
    echo "  name-register <name> <owner>      Register a new name"
    echo "  name-transfer <name> <new_owner>  Transfer name to new owner"
    echo "  name-update <name> <a_record>     Update name's A record"
    echo "  name-clear <name>                 Clear name's A record"
    echo "  name-protect <names...>           Bulk register protective names"
    echo ""
    echo -e "${GREEN}Transaction Commands:${NC}"
    echo "  tx-list [limit]                   List recent transactions"
    echo "  tx-from <address> [limit]         List transactions from address"
    echo "  tx-to <address> [limit]           List transactions to address"
    echo "  tx-add <from> <to> <value>        Add manual transaction"
    echo ""
    echo -e "${GREEN}Block Commands:${NC}"
    echo "  block-list [limit]                List recent blocks"
    echo "  block-by <address> [limit]        List blocks mined by address"
    echo "  block-stats                       Show mining statistics"
    echo ""
    echo -e "${GREEN}Statistics Commands:${NC}"
    echo "  stats                             Show overall statistics"
    echo "  rich-list [limit]                 Show richest addresses"
    echo "  name-stats                        Show name statistics"
    echo "  unpaid-stats                      Show unpaid mining pool stats"
    echo ""
    echo -e "${GREEN}Maintenance Commands:${NC}"
    echo "  purge-zero                        Remove unused zero-balance addresses"
    echo "  recalc-balance <address>          Recalculate balance from transactions"
    echo "  orphan-names                      Find names with non-existent owners"
    echo "  duplicate-check                   Check for duplicate entries"
    echo "  network-health                    Overall network health check"
    echo ""
    echo -e "${GREEN}Search Commands:${NC}"
    echo "  find-address <term>               Search addresses by partial match"
    echo "  find-name <term>                  Search names by partial match"
    echo "  active-miners [hours]             Show recently active miners"
    echo "  whale-watch [threshold]           Monitor large transactions"
    echo ""
    echo -e "${GREEN}Utility Commands:${NC}"
    echo "  supply-check                      Verify total supply calculations"
    echo "  backup [file]                     Backup database"
    echo "  sql <query>                       Execute raw SQL query"
    echo "  help                              Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 address-balance brvdgx5w3p 50000"
    echo "  $0 name-register admin brvdgx5w3p"
    echo "  $0 name-protect admin root system support"
    echo "  $0 rich-list 10"
}

# Address commands
case "$1" in
    "address-list")
        LIMIT=${2:-20}
        echo -e "${CYAN}Top $LIMIT addresses by balance:${NC}"
        query_sql "SELECT address, balance, totalin, totalout, firstseen, 
                   CASE WHEN locked = 1 THEN 'LOCKED' ELSE 'Active' END as status 
                   FROM addresses ORDER BY balance DESC LIMIT $LIMIT;"
        ;;
        
    "address-show")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Address required${NC}"
            exit 1
        fi
        echo -e "${CYAN}Address details for $2:${NC}"
        query_sql "SELECT a.*, 
                   (SELECT COUNT(*) FROM names WHERE owner = a.address) as names_owned,
                   (SELECT COUNT(*) FROM blocks WHERE address = a.address) as blocks_mined,
                   (SELECT SUM(value) FROM blocks WHERE address = a.address) as total_mined
                   FROM addresses a WHERE address = '$2';"
        echo -e "\n${CYAN}Names owned:${NC}"
        query_sql "SELECT name, unpaid, a, registered FROM names WHERE owner = '$2';"
        ;;
        
    "address-balance")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Address and balance required${NC}"
            exit 1
        fi
        exec_sql "UPDATE addresses SET balance = $3 WHERE address = '$2';"
        echo -e "${GREEN}✓ Balance updated for $2 to $3 BCN${NC}"
        ;;
        
    "address-add")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Address and balance required${NC}"
            exit 1
        fi
        exec_sql "INSERT INTO addresses (address, balance, totalin, totalout, firstseen) 
                  VALUES ('$2', $3, $3, 0, NOW());"
        echo -e "${GREEN}✓ Address $2 created with balance $3 BCN${NC}"
        ;;
        
    "address-lock")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Address required${NC}"
            exit 1
        fi
        exec_sql "UPDATE addresses SET locked = 1 WHERE address = '$2';"
        echo -e "${GREEN}✓ Address $2 locked${NC}"
        ;;
        
    "address-unlock")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Address required${NC}"
            exit 1
        fi
        exec_sql "UPDATE addresses SET locked = 0 WHERE address = '$2';"
        echo -e "${GREEN}✓ Address $2 unlocked${NC}"
        ;;
        
    "address-alert")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Address and message required${NC}"
            exit 1
        fi
        exec_sql "UPDATE addresses SET alert = '$3' WHERE address = '$2';"
        echo -e "${GREEN}✓ Alert set for $2${NC}"
        ;;

    # Name commands
    "name-list")
        if [ -z "$2" ]; then
            echo -e "${CYAN}All registered names:${NC}"
            query_sql "SELECT n.name, n.owner, a.balance as owner_balance, n.unpaid, n.a 
                       FROM names n 
                       JOIN addresses a ON n.owner = a.address 
                       ORDER BY n.name;"
        else
            echo -e "${CYAN}Names owned by $2:${NC}"
            query_sql "SELECT name, unpaid, a, registered, updated 
                       FROM names WHERE owner = '$2' ORDER BY name;"
        fi
        ;;
        
    "name-show")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Name required${NC}"
            exit 1
        fi
        echo -e "${CYAN}Details for name '$2':${NC}"
        query_sql "SELECT n.*, a.balance as owner_balance 
                   FROM names n 
                   JOIN addresses a ON n.owner = a.address 
                   WHERE n.name = '$2';"
        ;;
        
    "name-register")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Name and owner address required${NC}"
            exit 1
        fi
        exec_sql "INSERT INTO names (name, owner, original_owner, registered, updated, unpaid) 
                  VALUES ('$2', '$3', '$3', NOW(), NOW(), 500);"
        echo -e "${GREEN}✓ Name '$2' registered to $3${NC}"
        ;;
        
    "name-transfer")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Name and new owner required${NC}"
            exit 1
        fi
        exec_sql "UPDATE names SET owner = '$3', transferred = NOW(), updated = NOW() 
                  WHERE name = '$2';"
        echo -e "${GREEN}✓ Name '$2' transferred to $3${NC}"
        ;;
        
    "name-update")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Name and A record required${NC}"
            exit 1
        fi
        exec_sql "UPDATE names SET a = '$3', updated = NOW() WHERE name = '$2';"
        echo -e "${GREEN}✓ A record updated for '$2'${NC}"
        ;;
        
    "name-clear")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Name required${NC}"
            exit 1
        fi
        exec_sql "UPDATE names SET a = NULL, updated = NOW() WHERE name = '$2';"
        echo -e "${GREEN}✓ A record cleared for '$2'${NC}"
        ;;
        
    "name-protect")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: At least one name required${NC}"
            exit 1
        fi
        shift # Remove command from arguments
        OWNER="brvdgx5w3p" # Default protection address
        for name in "$@"; do
            exec_sql "INSERT IGNORE INTO names (name, owner, original_owner, registered, updated, unpaid) 
                      VALUES ('$name', '$OWNER', '$OWNER', NOW(), NOW(), 500);"
            echo -e "${GREEN}✓ Protected name '$name'${NC}"
        done
        ;;

    # Transaction commands
    "tx-list")
        LIMIT=${2:-20}
        echo -e "${CYAN}Recent $LIMIT transactions:${NC}"
        query_sql "SELECT id, \`from\`, \`to\`, value, time, name, 
                   SUBSTRING(op, 1, 30) as metadata 
                   FROM transactions ORDER BY time DESC LIMIT $LIMIT;"
        ;;
        
    "tx-from")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Address required${NC}"
            exit 1
        fi
        LIMIT=${3:-20}
        echo -e "${CYAN}Transactions from $2:${NC}"
        query_sql "SELECT id, \`to\`, value, time, name 
                   FROM transactions WHERE \`from\` = '$2' 
                   ORDER BY time DESC LIMIT $LIMIT;"
        ;;
        
    "tx-to")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Address required${NC}"
            exit 1
        fi
        LIMIT=${3:-20}
        echo -e "${CYAN}Transactions to $2:${NC}"
        query_sql "SELECT id, \`from\`, value, time, name 
                   FROM transactions WHERE \`to\` = '$2' 
                   ORDER BY time DESC LIMIT $LIMIT;"
        ;;
        
    "tx-add")
        if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
            echo -e "${RED}Error: From, to, and value required${NC}"
            exit 1
        fi
        exec_sql "INSERT INTO transactions (\`from\`, \`to\`, value, time) 
                  VALUES ('$2', '$3', $4, NOW());"
        # Update balances
        exec_sql "UPDATE addresses SET balance = balance - $4, totalout = totalout + $4 
                  WHERE address = '$2';"
        exec_sql "UPDATE addresses SET balance = balance + $4, totalin = totalin + $4 
                  WHERE address = '$3';"
        echo -e "${GREEN}✓ Transaction added: $2 → $3 ($4 BCN)${NC}"
        ;;

    # Block commands
    "block-list")
        LIMIT=${2:-20}
        echo -e "${CYAN}Recent $LIMIT blocks:${NC}"
        query_sql "SELECT id as height, address, value, 
                   SUBSTRING(hash, 1, 12) as hash_start, difficulty, time 
                   FROM blocks ORDER BY id DESC LIMIT $LIMIT;"
        ;;
        
    "block-by")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Address required${NC}"
            exit 1
        fi
        LIMIT=${3:-20}
        echo -e "${CYAN}Blocks mined by $2:${NC}"
        query_sql "SELECT id as height, value, 
                   SUBSTRING(hash, 1, 12) as hash_start, difficulty, time 
                   FROM blocks WHERE address = '$2' 
                   ORDER BY id DESC LIMIT $LIMIT;"
        ;;
        
    "block-stats")
        echo -e "${CYAN}Mining statistics:${NC}"
        query_sql "SELECT 
                   COUNT(*) as total_blocks,
                   COUNT(DISTINCT address) as unique_miners,
                   SUM(value) as total_rewards,
                   AVG(difficulty) as avg_difficulty,
                   MAX(id) as latest_block
                   FROM blocks;"
        echo -e "\n${CYAN}Top 10 miners:${NC}"
        query_sql "SELECT address, COUNT(*) as blocks_mined, 
                   SUM(value) as total_earned 
                   FROM blocks GROUP BY address 
                   ORDER BY blocks_mined DESC LIMIT 10;"
        ;;

    # Statistics commands
    "stats")
        echo -e "${CYAN}Bckn Network Statistics:${NC}"
        query_sql "SELECT 
                   (SELECT COUNT(*) FROM addresses) as total_addresses,
                   (SELECT COUNT(*) FROM addresses WHERE balance > 0) as active_addresses,
                   (SELECT SUM(balance) FROM addresses) as total_supply,
                   (SELECT COUNT(*) FROM names) as total_names,
                   (SELECT COUNT(*) FROM transactions) as total_transactions,
                   (SELECT MAX(id) FROM blocks) as total_blocks;"
        ;;
        
    "rich-list")
        LIMIT=${2:-20}
        echo -e "${CYAN}Top $LIMIT richest addresses:${NC}"
        query_sql "SELECT a.address, a.balance, 
                   (SELECT COUNT(*) FROM names WHERE owner = a.address) as names,
                   (SELECT COUNT(*) FROM blocks WHERE address = a.address) as blocks,
                   CASE WHEN a.locked = 1 THEN 'LOCKED' ELSE '' END as status
                   FROM addresses a 
                   WHERE a.balance > 0 
                   ORDER BY a.balance DESC LIMIT $LIMIT;"
        ;;
        
    "name-stats")
        echo -e "${CYAN}Name system statistics:${NC}"
        query_sql "SELECT 
                   COUNT(*) as total_names,
                   COUNT(DISTINCT owner) as unique_owners,
                   SUM(CAST(unpaid AS UNSIGNED)) as total_unpaid,
                   COUNT(CASE WHEN unpaid > 0 THEN 1 END) as names_with_unpaid
                   FROM names;"
        echo -e "\n${CYAN}Top name holders:${NC}"
        query_sql "SELECT owner, COUNT(*) as names_owned, 
                   SUM(CAST(unpaid AS UNSIGNED)) as total_unpaid 
                   FROM names GROUP BY owner 
                   ORDER BY names_owned DESC LIMIT 10;"
        ;;
        
    "unpaid-stats")
        echo -e "${CYAN}Unpaid mining pool statistics:${NC}"
        query_sql "SELECT 
                   SUM(CAST(unpaid AS UNSIGNED)) as total_unpaid_pool,
                   COUNT(CASE WHEN unpaid > 0 THEN 1 END) as names_contributing,
                   MAX(CAST(unpaid AS UNSIGNED)) as max_unpaid,
                   AVG(CAST(unpaid AS UNSIGNED)) as avg_unpaid
                   FROM names WHERE unpaid > 0;"
        echo -e "\n${CYAN}Names with highest unpaid:${NC}"
        query_sql "SELECT name, owner, unpaid 
                   FROM names 
                   WHERE unpaid > 0 
                   ORDER BY CAST(unpaid AS UNSIGNED) DESC LIMIT 10;"
        ;;

    # Maintenance commands
    "purge-zero")
        echo -e "${CYAN}Addresses with 0 balance:${NC}"
        query_sql "SELECT COUNT(*) as count FROM addresses WHERE balance = 0;"
        read -p "Delete all zero-balance addresses with no transactions? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            exec_sql "DELETE FROM addresses WHERE balance = 0 
                      AND id NOT IN (SELECT DISTINCT \`from\` FROM transactions WHERE \`from\` IS NOT NULL)
                      AND id NOT IN (SELECT DISTINCT \`to\` FROM transactions WHERE \`to\` IS NOT NULL)
                      AND address NOT IN (SELECT DISTINCT owner FROM names)
                      AND address NOT IN (SELECT DISTINCT address FROM blocks);"
            echo -e "${GREEN}✓ Purged unused zero-balance addresses${NC}"
        fi
        ;;
        
    "recalc-balance")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Address required${NC}"
            exit 1
        fi
        echo -e "${CYAN}Recalculating balance for $2...${NC}"
        query_sql "SELECT 
                   (SELECT COALESCE(SUM(value), 0) FROM transactions WHERE \`to\` = '$2') as total_in,
                   (SELECT COALESCE(SUM(value), 0) FROM transactions WHERE \`from\` = '$2') as total_out,
                   (SELECT COALESCE(SUM(value), 0) FROM blocks WHERE address = '$2') as mined;"
        exec_sql "UPDATE addresses SET 
                  totalin = (SELECT COALESCE(SUM(value), 0) FROM transactions WHERE \`to\` = '$2') +
                           (SELECT COALESCE(SUM(value), 0) FROM blocks WHERE address = '$2'),
                  totalout = (SELECT COALESCE(SUM(value), 0) FROM transactions WHERE \`from\` = '$2'),
                  balance = totalin - totalout
                  WHERE address = '$2';"
        echo -e "${GREEN}✓ Balance recalculated for $2${NC}"
        ;;
        
    "find-address")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Search term required${NC}"
            exit 1
        fi
        echo -e "${CYAN}Searching for addresses containing '$2':${NC}"
        query_sql "SELECT address, balance, 
                   (SELECT COUNT(*) FROM names WHERE owner = addresses.address) as names
                   FROM addresses 
                   WHERE address LIKE '%$2%' 
                   ORDER BY balance DESC LIMIT 20;"
        ;;
        
    "find-name")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Search term required${NC}"
            exit 1
        fi
        echo -e "${CYAN}Searching for names containing '$2':${NC}"
        query_sql "SELECT name, owner, a, unpaid 
                   FROM names 
                   WHERE name LIKE '%$2%' 
                   ORDER BY name LIMIT 20;"
        ;;
        
    "orphan-names")
        echo -e "${CYAN}Names with non-existent owners:${NC}"
        query_sql "SELECT n.name, n.owner, n.unpaid 
                   FROM names n 
                   LEFT JOIN addresses a ON n.owner = a.address 
                   WHERE a.address IS NULL;"
        ;;
        
    "duplicate-check")
        echo -e "${CYAN}Checking for duplicate issues...${NC}"
        echo -e "\n${YELLOW}Duplicate transactions:${NC}"
        query_sql "SELECT \`from\`, \`to\`, value, time, COUNT(*) as count 
                   FROM transactions 
                   GROUP BY \`from\`, \`to\`, value, time 
                   HAVING count > 1 LIMIT 10;"
        echo -e "\n${YELLOW}Duplicate block hashes:${NC}"
        query_sql "SELECT hash, COUNT(*) as count 
                   FROM blocks 
                   GROUP BY hash 
                   HAVING count > 1 LIMIT 10;"
        ;;
        
    "active-miners")
        HOURS=${2:-24}
        echo -e "${CYAN}Miners active in last $HOURS hours:${NC}"
        query_sql "SELECT b.address, COUNT(*) as blocks_found, 
                   SUM(b.value) as earned, MAX(b.time) as last_block 
                   FROM blocks b 
                   WHERE b.time > DATE_SUB(NOW(), INTERVAL $HOURS HOUR) 
                   GROUP BY b.address 
                   ORDER BY blocks_found DESC;"
        ;;
        
    "whale-watch")
        THRESHOLD=${2:-100000}
        echo -e "${CYAN}Whale movements (transactions >= $THRESHOLD BCN):${NC}"
        query_sql "SELECT t.id, t.\`from\`, t.\`to\`, t.value, t.time 
                   FROM transactions t 
                   WHERE t.value >= $THRESHOLD 
                   ORDER BY t.time DESC LIMIT 20;"
        ;;
        
    "supply-check")
        echo -e "${CYAN}Supply verification:${NC}"
        query_sql "SELECT 
                   (SELECT SUM(balance) FROM addresses) as total_in_addresses,
                   (SELECT SUM(value) FROM blocks) as total_mined,
                   (SELECT SUM(CAST(unpaid AS UNSIGNED)) FROM names) as total_unpaid,
                   (SELECT SUM(value) FROM transactions WHERE \`to\` = 'name') as total_name_costs;"
        ;;
        
    "network-health")
        echo -e "${CYAN}Network Health Check:${NC}"
        query_sql "SELECT 
                   (SELECT COUNT(*) FROM blocks WHERE time > DATE_SUB(NOW(), INTERVAL 1 HOUR)) as blocks_last_hour,
                   (SELECT COUNT(*) FROM transactions WHERE time > DATE_SUB(NOW(), INTERVAL 1 HOUR)) as tx_last_hour,
                   (SELECT COUNT(DISTINCT address) FROM blocks WHERE time > DATE_SUB(NOW(), INTERVAL 24 HOUR)) as unique_miners_24h,
                   (SELECT MAX(time) FROM blocks) as last_block_time,
                   (SELECT AVG(difficulty) FROM blocks WHERE id > (SELECT MAX(id) - 100 FROM blocks)) as avg_recent_difficulty;"
        ;;

    # Utility commands
    "backup")
        BACKUP_FILE=${2:-"bckn_backup_$(date +%Y%m%d_%H%M%S).sql"}
        mysqldump -h $DB_HOST -u $DB_USER ${DB_PASS:+-p$DB_PASS} $DB_NAME > $BACKUP_FILE
        echo -e "${GREEN}✓ Database backed up to $BACKUP_FILE${NC}"
        ;;
        
    "sql")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: SQL query required${NC}"
            exit 1
        fi
        shift # Remove command
        QUERY="$*"
        echo -e "${CYAN}Executing SQL:${NC} $QUERY"
        query_sql "$QUERY"
        ;;
        
    "help"|"--help"|"-h"|"")
        show_help
        ;;
        
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' to see available commands"
        exit 1
        ;;
esac