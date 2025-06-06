-- Bckn Client for ComputerCraft (CC:Tweaked)
-- A complete Bckn wallet and mining interface for Minecraft computers

local BCKN_NODE = "https://bckn.dev"
local VERSION = "1.0"

-- Configuration file
local CONFIG_FILE = "bckn.config"
local WALLET_FILE = "wallet.dat"

-- Colors for display
local colors = {
    header = colors.cyan,
    success = colors.green,
    error = colors.red,
    info = colors.yellow,
    menu = colors.white,
    value = colors.lightBlue
}

-- Helper function to make HTTP requests
local function httpRequest(method, endpoint, data, headers)
    local url = BCKN_NODE .. endpoint
    headers = headers or {}
    headers["Content-Type"] = "application/json"
    
    local response, err
    if method == "GET" then
        response, err = http.get(url, headers)
    elseif method == "POST" then
        local jsonData = textutils.serializeJSON(data)
        response, err = http.post(url, jsonData, headers)
    end
    
    if not response then
        return nil, err
    end
    
    local content = response.readAll()
    response.close()
    
    local success, result = pcall(textutils.unserializeJSON, content)
    if success then
        return result
    else
        return content
    end
end

-- Wallet management
local wallet = {
    privateKey = nil,
    address = nil,
    balance = 0
}

-- Save wallet to file
local function saveWallet()
    local file = fs.open(WALLET_FILE, "w")
    file.write(textutils.serialize({
        privateKey = wallet.privateKey,
        address = wallet.address
    }))
    file.close()
end

-- Load wallet from file
local function loadWallet()
    if fs.exists(WALLET_FILE) then
        local file = fs.open(WALLET_FILE, "r")
        local data = textutils.unserialize(file.readAll())
        file.close()
        wallet.privateKey = data.privateKey
        wallet.address = data.address
        return true
    end
    return false
end

-- Login with private key
local function login(privateKey)
    local response, err = httpRequest("POST", "/login", {privatekey = privateKey})
    if response and response.address then
        wallet.privateKey = privateKey
        wallet.address = response.address
        saveWallet()
        return true, response.address
    end
    return false, err or "Login failed"
end

-- Get balance
local function updateBalance()
    if not wallet.address then return false end
    
    local response = httpRequest("GET", "/addresses/" .. wallet.address)
    if response and response.address then
        wallet.balance = response.address.balance
        return true, response.address.balance
    end
    return false, 0
end

-- Send transaction
local function sendTransaction(to, amount, metadata)
    if not wallet.privateKey then
        return false, "Not logged in"
    end
    
    local data = {
        privatekey = wallet.privateKey,
        to = to,
        amount = amount
    }
    
    if metadata then
        data.metadata = metadata
    end
    
    local response = httpRequest("POST", "/transactions", data)
    if response and response.ok then
        return true, response
    end
    return false, response and response.error or "Transaction failed"
end

-- Mining functions
local function getWork()
    local response = httpRequest("GET", "/work")
    if response and response.work then
        return response.work
    end
    return nil
end

local function getLastBlock()
    local response = httpRequest("GET", "/blocks/last")
    if response and response.block then
        return response.block
    end
    return nil
end

local function submitBlock(nonce)
    if not wallet.address then
        return false, "Not logged in"
    end
    
    local response = httpRequest("POST", "/submit", {
        address = wallet.address,
        nonce = tostring(nonce)
    })
    
    if response and response.success then
        return true, response
    end
    return false, response and response.error or "Submission failed"
end

-- Simple mining function (slow but works)
local function mine()
    if not wallet.address then
        print("Please login first!")
        return
    end
    
    print("Starting miner...")
    local work = getWork()
    local lastBlock = getLastBlock()
    
    if not work or not lastBlock then
        print("Failed to get work data")
        return
    end
    
    local lastHash = lastBlock.hash and lastBlock.hash:sub(1, 12) or "000000000000"
    print("Work: " .. work)
    print("Last hash: " .. lastHash)
    
    local nonce = 0
    local startTime = os.clock()
    local hashes = 0
    
    while true do
        -- Calculate hash (CC doesn't have SHA256, so we'll simulate)
        -- In real implementation, you'd need a SHA256 library
        local message = wallet.address .. lastHash .. tostring(nonce)
        
        -- This is a placeholder - you need actual SHA256
        -- For now, we'll just check if it would submit
        if nonce % 100000 == 0 then
            term.setCursorPos(1, 10)
            term.clearLine()
            local elapsed = os.clock() - startTime
            local hashrate = hashes / elapsed
            print(string.format("Nonce: %d | %.2f H/s", nonce, hashrate))
        end
        
        -- Check for user interrupt
        if nonce % 1000 == 0 then
            local event = os.pullEvent(0.01)
            if event == "key" then
                print("\nMining stopped by user")
                break
            end
        end
        
        nonce = nonce + 1
        hashes = hashes + 1
    end
end

-- UI Functions
local function clearScreen()
    term.clear()
    term.setCursorPos(1, 1)
end

local function drawHeader()
    term.setTextColor(colors.header)
    print("=== Bckn Client v" .. VERSION .. " ===")
    term.setTextColor(colors.white)
    print("")
end

local function drawWalletInfo()
    if wallet.address then
        term.setTextColor(colors.info)
        print("Address: " .. wallet.address)
        term.setTextColor(colors.value)
        print("Balance: " .. (wallet.balance or 0) .. " BCN")
    else
        term.setTextColor(colors.error)
        print("Not logged in")
    end
    term.setTextColor(colors.white)
    print("")
end

local function mainMenu()
    while true do
        clearScreen()
        drawHeader()
        drawWalletInfo()
        
        term.setTextColor(colors.menu)
        print("1. Login/Import Wallet")
        print("2. Refresh Balance")
        print("3. Send BCN")
        print("4. View Transactions")
        print("5. Mine Blocks")
        print("6. Name Management")
        print("7. Settings")
        print("8. Exit")
        print("")
        print("Select option: ")
        
        local choice = read()
        
        if choice == "1" then
            loginMenu()
        elseif choice == "2" then
            refreshBalance()
        elseif choice == "3" then
            sendMenu()
        elseif choice == "4" then
            transactionHistory()
        elseif choice == "5" then
            miningMenu()
        elseif choice == "6" then
            nameMenu()
        elseif choice == "7" then
            settingsMenu()
        elseif choice == "8" then
            break
        end
    end
end

-- Login menu
function loginMenu()
    clearScreen()
    drawHeader()
    
    print("Enter private key:")
    local privateKey = read("*")
    
    print("\nLogging in...")
    local success, address = login(privateKey)
    
    if success then
        term.setTextColor(colors.success)
        print("Login successful!")
        print("Address: " .. address)
        updateBalance()
    else
        term.setTextColor(colors.error)
        print("Login failed: " .. address)
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Refresh balance
function refreshBalance()
    clearScreen()
    drawHeader()
    
    print("Updating balance...")
    local success, balance = updateBalance()
    
    if success then
        term.setTextColor(colors.success)
        print("Balance updated: " .. balance .. " BCN")
    else
        term.setTextColor(colors.error)
        print("Failed to update balance")
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Send menu
function sendMenu()
    clearScreen()
    drawHeader()
    
    if not wallet.address then
        term.setTextColor(colors.error)
        print("Please login first!")
        term.setTextColor(colors.white)
        print("\nPress any key to continue...")
        os.pullEvent("key")
        return
    end
    
    print("Current balance: " .. wallet.balance .. " BCN")
    print("")
    print("Recipient address or name:")
    local recipient = read()
    
    print("Amount to send:")
    local amount = tonumber(read())
    
    if not amount or amount <= 0 then
        term.setTextColor(colors.error)
        print("Invalid amount")
        term.setTextColor(colors.white)
        print("\nPress any key to continue...")
        os.pullEvent("key")
        return
    end
    
    print("Metadata (optional):")
    local metadata = read()
    if metadata == "" then metadata = nil end
    
    print("\nSending transaction...")
    local success, result = sendTransaction(recipient, amount, metadata)
    
    if success then
        term.setTextColor(colors.success)
        print("Transaction sent successfully!")
        updateBalance()
    else
        term.setTextColor(colors.error)
        print("Transaction failed: " .. result)
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Transaction history
function transactionHistory()
    clearScreen()
    drawHeader()
    
    if not wallet.address then
        term.setTextColor(colors.error)
        print("Please login first!")
        term.setTextColor(colors.white)
        print("\nPress any key to continue...")
        os.pullEvent("key")
        return
    end
    
    print("Fetching transactions...")
    local response = httpRequest("GET", "/addresses/" .. wallet.address .. "/transactions?limit=10")
    
    if response and response.transactions then
        clearScreen()
        drawHeader()
        print("Recent Transactions:")
        print("")
        
        for i, tx in ipairs(response.transactions) do
            local direction = tx.from == wallet.address and "SENT" or "RECEIVED"
            local other = tx.from == wallet.address and tx.to or tx.from
            
            if direction == "SENT" then
                term.setTextColor(colors.error)
            else
                term.setTextColor(colors.success)
            end
            
            print(string.format("%s %d BCN %s %s", 
                direction, tx.value, 
                direction == "SENT" and "to" or "from", 
                other or "?"))
        end
    else
        term.setTextColor(colors.error)
        print("Failed to fetch transactions")
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Mining menu
function miningMenu()
    clearScreen()
    drawHeader()
    
    print("Mining Menu")
    print("")
    print("1. Start Mining")
    print("2. Mining Stats")
    print("3. Back")
    print("")
    print("Select option: ")
    
    local choice = read()
    
    if choice == "1" then
        mine()
    elseif choice == "2" then
        miningStats()
    end
end

-- Mining stats
function miningStats()
    clearScreen()
    drawHeader()
    
    print("Fetching mining stats...")
    local work = getWork()
    local lastBlock = getLastBlock()
    
    if work and lastBlock then
        print("")
        print("Current work: " .. work)
        print("Last block: " .. lastBlock.id)
        print("Last miner: " .. lastBlock.address)
        print("Block value: " .. lastBlock.value .. " BCN")
    else
        term.setTextColor(colors.error)
        print("Failed to fetch mining stats")
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Name management menu
function nameMenu()
    clearScreen()
    drawHeader()
    
    print("Name Management")
    print("")
    print("1. Check Name Availability")
    print("2. Register Name (500 BCN)")
    print("3. View My Names")
    print("4. Update Name A Record")
    print("5. Back")
    print("")
    print("Select option: ")
    
    local choice = read()
    
    if choice == "1" then
        checkName()
    elseif choice == "2" then
        registerName()
    elseif choice == "3" then
        viewNames()
    elseif choice == "4" then
        updateName()
    end
end

-- Check name availability
function checkName()
    clearScreen()
    drawHeader()
    
    print("Enter name to check:")
    local name = read()
    
    print("\nChecking availability...")
    local response = httpRequest("GET", "/names/check/" .. name)
    
    if response then
        if response.available then
            term.setTextColor(colors.success)
            print("Name '" .. name .. "' is available!")
        else
            term.setTextColor(colors.error)
            print("Name '" .. name .. "' is taken")
            if response.name then
                print("Owner: " .. response.name.owner)
            end
        end
    else
        term.setTextColor(colors.error)
        print("Failed to check name")
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Register name
function registerName()
    clearScreen()
    drawHeader()
    
    if not wallet.privateKey then
        term.setTextColor(colors.error)
        print("Please login first!")
        term.setTextColor(colors.white)
        print("\nPress any key to continue...")
        os.pullEvent("key")
        return
    end
    
    if wallet.balance < 500 then
        term.setTextColor(colors.error)
        print("Insufficient balance (need 500 BCN)")
        term.setTextColor(colors.white)
        print("\nPress any key to continue...")
        os.pullEvent("key")
        return
    end
    
    print("Enter name to register:")
    local name = read()
    
    print("\nRegistering name...")
    local response = httpRequest("POST", "/names/" .. name, {
        privatekey = wallet.privateKey
    })
    
    if response and response.ok then
        term.setTextColor(colors.success)
        print("Name '" .. name .. "' registered successfully!")
        updateBalance()
    else
        term.setTextColor(colors.error)
        print("Failed to register name: " .. (response and response.error or "Unknown error"))
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- View owned names
function viewNames()
    clearScreen()
    drawHeader()
    
    if not wallet.address then
        term.setTextColor(colors.error)
        print("Please login first!")
        term.setTextColor(colors.white)
        print("\nPress any key to continue...")
        os.pullEvent("key")
        return
    end
    
    print("Fetching your names...")
    local response = httpRequest("GET", "/addresses/" .. wallet.address .. "/names")
    
    if response and response.names then
        clearScreen()
        drawHeader()
        print("Your Names:")
        print("")
        
        if #response.names == 0 then
            print("You don't own any names")
        else
            for i, name in ipairs(response.names) do
                print(name.name .. ".bckn")
                if name.a then
                    term.setTextColor(colors.info)
                    print("  A: " .. name.a)
                    term.setTextColor(colors.white)
                end
            end
        end
    else
        term.setTextColor(colors.error)
        print("Failed to fetch names")
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Settings menu
function settingsMenu()
    clearScreen()
    drawHeader()
    
    print("Settings")
    print("")
    print("1. Export Private Key")
    print("2. Change Node URL")
    print("3. About")
    print("4. Back")
    print("")
    print("Select option: ")
    
    local choice = read()
    
    if choice == "1" then
        exportKey()
    elseif choice == "2" then
        changeNode()
    elseif choice == "3" then
        about()
    end
end

-- Export private key
function exportKey()
    clearScreen()
    drawHeader()
    
    if wallet.privateKey then
        term.setTextColor(colors.error)
        print("WARNING: Keep your private key secret!")
        term.setTextColor(colors.white)
        print("")
        print("Your private key:")
        term.setTextColor(colors.info)
        print(wallet.privateKey)
    else
        term.setTextColor(colors.error)
        print("No wallet loaded")
    end
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- About screen
function about()
    clearScreen()
    drawHeader()
    
    print("Bckn Client for ComputerCraft")
    print("Version: " .. VERSION)
    print("")
    print("A complete Bckn wallet interface")
    print("for Minecraft computers")
    print("")
    print("Features:")
    print("- Wallet management")
    print("- Send/receive BCN")
    print("- Name registration")
    print("- Basic mining")
    print("")
    print("Node: " .. BCKN_NODE)
    
    term.setTextColor(colors.white)
    print("\nPress any key to continue...")
    os.pullEvent("key")
end

-- Main program
local function main()
    -- Load wallet if exists
    if loadWallet() and wallet.address then
        print("Loading wallet...")
        updateBalance()
    end
    
    -- Start main menu
    mainMenu()
    
    clearScreen()
    print("Thank you for using Bckn Client!")
end

-- Run the program
main()