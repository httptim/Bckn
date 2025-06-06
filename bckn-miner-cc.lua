-- Bckn Miner for ComputerCraft with SHA256 Implementation
-- Based on Krist mining algorithms adapted for Bckn

-- SHA256 implementation for ComputerCraft
-- Adapted from pure Lua SHA256 implementations
local sha256 = {}

-- SHA256 constants
local k = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
}

-- Bitwise operations for CC (using bit API)
local band = bit32 and bit32.band or bit.band
local bor = bit32 and bit32.bor or bit.bor
local bxor = bit32 and bit32.bxor or bit.bxor
local bnot = bit32 and bit32.bnot or bit.bnot
local rshift = bit32 and bit32.rshift or bit.brshift
local lshift = bit32 and bit32.lshift or bit.blshift

local function rrotate(n, b)
    return bor(rshift(n, b), lshift(n, 32 - b))
end

-- Convert string to big-endian 32-bit words
local function str2word(str)
    local words = {}
    for i = 1, #str, 4 do
        local w = 0
        for j = 0, 3 do
            w = w * 256 + (str:byte(i + j) or 0)
        end
        words[#words + 1] = w
    end
    return words
end

-- Convert words to hex string
local function words2hex(words)
    local hex = ""
    for i = 1, #words do
        hex = hex .. string.format("%08x", words[i])
    end
    return hex
end

-- SHA256 compression function
local function sha256_compress(H, words)
    local a, b, c, d, e, f, g, h = H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]
    local W = {}
    
    -- Initialize working variables
    for i = 1, 16 do
        W[i] = words[i] or 0
    end
    
    for i = 17, 64 do
        local s0 = bxor(rrotate(W[i-15], 7), rrotate(W[i-15], 18), rshift(W[i-15], 3))
        local s1 = bxor(rrotate(W[i-2], 17), rrotate(W[i-2], 19), rshift(W[i-2], 10))
        W[i] = band(W[i-16] + s0 + W[i-7] + s1, 0xffffffff)
    end
    
    -- Compression
    for i = 1, 64 do
        local S1 = bxor(rrotate(e, 6), rrotate(e, 11), rrotate(e, 25))
        local ch = bxor(band(e, f), band(bnot(e), g))
        local temp1 = band(h + S1 + ch + k[i] + W[i], 0xffffffff)
        local S0 = bxor(rrotate(a, 2), rrotate(a, 13), rrotate(a, 22))
        local maj = bxor(band(a, b), band(a, c), band(b, c))
        local temp2 = band(S0 + maj, 0xffffffff)
        
        h = g
        g = f
        f = e
        e = band(d + temp1, 0xffffffff)
        d = c
        c = b
        b = a
        a = band(temp1 + temp2, 0xffffffff)
    end
    
    H[1] = band(H[1] + a, 0xffffffff)
    H[2] = band(H[2] + b, 0xffffffff)
    H[3] = band(H[3] + c, 0xffffffff)
    H[4] = band(H[4] + d, 0xffffffff)
    H[5] = band(H[5] + e, 0xffffffff)
    H[6] = band(H[6] + f, 0xffffffff)
    H[7] = band(H[7] + g, 0xffffffff)
    H[8] = band(H[8] + h, 0xffffffff)
end

-- Main SHA256 function
function sha256.hash(msg)
    -- Initial hash values
    local H = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    }
    
    -- Preprocessing
    local msgLen = #msg
    local padLen = (55 - msgLen) % 64
    local padded = msg .. "\128" .. string.rep("\0", padLen) .. string.char(
        0, 0, 0, 0,
        band(rshift(msgLen * 8, 24), 0xff),
        band(rshift(msgLen * 8, 16), 0xff),
        band(rshift(msgLen * 8, 8), 0xff),
        band(msgLen * 8, 0xff)
    )
    
    -- Process message in 512-bit chunks
    for i = 1, #padded, 64 do
        local chunk = padded:sub(i, i + 63)
        local words = str2word(chunk)
        sha256_compress(H, words)
    end
    
    return words2hex(H)
end

-- Bckn Miner Implementation
local BCKN_NODE = "https://bckn.dev"

-- Helper function for HTTP requests
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

-- Mining statistics
local stats = {
    startTime = 0,
    hashes = 0,
    blocksFound = 0,
    lastHashrate = 0
}

-- Get work from server
local function getWork()
    local response = httpRequest("GET", "/work")
    if response and response.work then
        return response.work
    end
    return nil
end

-- Get last block
local function getLastBlock()
    local response = httpRequest("GET", "/blocks/last")
    if response and response.block then
        return response.block
    end
    return nil
end

-- Submit nonce
local function submitNonce(address, nonce)
    local response = httpRequest("POST", "/submit", {
        address = address,
        nonce = tostring(nonce)
    })
    return response and response.success
end

-- Mining function
local function mine(privateKey)
    print("Logging in...")
    local loginResp = httpRequest("POST", "/login", {privatekey = privateKey})
    if not loginResp or not loginResp.address then
        print("Login failed!")
        return
    end
    
    local address = loginResp.address
    print("Mining address: " .. address)
    
    while true do
        -- Get current work
        local work = getWork()
        local lastBlock = getLastBlock()
        
        if not work or not lastBlock then
            print("Failed to get work data")
            sleep(5)
        else
            local lastHash = lastBlock.hash and lastBlock.hash:sub(1, 12) or "000000000000"
            local prefix = address .. lastHash
            
            -- Initial display will be drawn by the mining loop
            
            stats.startTime = os.clock()
            stats.hashes = 0
            
            -- Start mining with random nonce
            local nonce = math.random(0, 2147483647)
            local found = false
            
            -- Add debug mode flag
            local debugMode = true  -- Set to false to disable debug output
            local debugInterval = 500  -- Update display every N hashes
            
            -- Store recent hashes for display
            local recentHashes = {}
            local maxRecent = 8
            
            while not found do
                -- Calculate hash
                local message = prefix .. tostring(nonce)
                local hash = sha256.hash(message)
                local hashValue = tonumber(hash:sub(1, 12), 16)
                
                stats.hashes = stats.hashes + 1
                
                -- Update display
                if debugMode and stats.hashes % debugInterval == 0 then
                    -- Add to recent hashes
                    table.insert(recentHashes, {
                        nonce = nonce,
                        hash = hash:sub(1, 12),
                        value = hashValue,
                        count = stats.hashes
                    })
                    if #recentHashes > maxRecent then
                        table.remove(recentHashes, 1)
                    end
                    
                    -- Clear screen and redraw
                    term.clear()
                    term.setCursorPos(1, 1)
                    print("=== Bckn Miner for ComputerCraft ===")
                    print("Address: " .. address)
                    print("Work: " .. work .. " | Last: " .. lastHash)
                    
                    -- Stats
                    local elapsed = os.clock() - stats.startTime
                    local hashrate = stats.hashes / elapsed
                    print(string.format("Rate: %.1f H/s | Total: %d | Blocks: %d", 
                        hashrate, stats.hashes, stats.blocksFound))
                    print("")
                    print("Recent Mining Attempts:")
                    print("------------------------")
                    
                    -- Show recent hashes
                    for i, h in ipairs(recentHashes) do
                        local marker = h.value <= work and ">>>" or "   "
                        print(string.format("%s[%d] %s = %d", 
                            marker, h.count, h.hash, h.value))
                    end
                    
                    print("")
                    print("Press Ctrl+T to stop mining")
                    
                    -- Yield to update display
                    os.queueEvent("mining")
                    os.pullEvent("mining")
                end
                
                -- Check if valid
                if hashValue <= work then
                    term.clear()
                    term.setCursorPos(1, 1)
                    print("=== FOUND VALID NONCE! ===")
                    print("")
                    print("Nonce: " .. nonce)
                    print("Message: " .. message:sub(1, 50) .. "...")
                    print("Hash: " .. hash)
                    print("Hash Value: " .. hashValue .. " <= " .. work)
                    print("")
                    print("Submitting...")
                    
                    if submitNonce(address, nonce) then
                        stats.blocksFound = stats.blocksFound + 1
                        print("Block submitted successfully!")
                        print("Total blocks found: " .. stats.blocksFound)
                        found = true
                        sleep(5)  -- Give time to read success
                    else
                        print("Submission failed, continuing...")
                    end
                end
                
                -- Yield regularly to keep display responsive
                if stats.hashes % 50 == 0 then
                    sleep(0)  -- Tiny sleep to prevent "too long without yielding"
                end
                
                nonce = nonce + 1
            end
        end
    end
end

-- Simple UI for standalone miner
local function main()
    term.clear()
    term.setCursorPos(1, 1)
    print("=== Bckn Miner for ComputerCraft ===")
    print("")
    print("1. Start Mining")
    print("2. Test SHA256")
    print("3. Exit")
    print("")
    print("Select option: ")
    
    local choice = read()
    
    if choice == "1" then
        print("\nEnter private key:")
        local privateKey = read("*")
        mine(privateKey)
    elseif choice == "2" then
        print("\nTesting SHA256 Implementation...")
        print("=====================================")
        
        -- Test 1: Basic hash
        local test1 = sha256.hash("hello")
        print("\nTest 1 - Basic:")
        print("Input: 'hello'")
        print("Output: " .. test1)
        print("Expected: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
        print("Match: " .. (test1 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" and "YES" or "NO"))
        
        -- Test 2: Simulate mining
        print("\nTest 2 - Mining Simulation:")
        local testAddress = "k5znkjvnmn"
        local testLastHash = "000000000abc"
        local testNonce = "12345"
        local testMessage = testAddress .. testLastHash .. testNonce
        local testHash = sha256.hash(testMessage)
        
        print("Address: " .. testAddress)
        print("Last Hash: " .. testLastHash)
        print("Nonce: " .. testNonce)
        print("Message: " .. testMessage)
        print("Hash: " .. testHash)
        print("First 12: " .. testHash:sub(1, 12))
        print("As number: " .. tonumber(testHash:sub(1, 12), 16))
        
        -- Test 3: Performance test
        print("\nTest 3 - Performance:")
        local startTime = os.clock()
        local testHashes = 100
        for i = 1, testHashes do
            sha256.hash("test" .. i)
        end
        local elapsed = os.clock() - startTime
        print(string.format("Computed %d hashes in %.3f seconds", testHashes, elapsed))
        print(string.format("Rate: %.1f H/s", testHashes / elapsed))
        
        print("\n\nPress any key to continue...")
        os.pullEvent("key")
        main()
    elseif choice == "3" then
        print("Goodbye!")
    else
        main()
    end
end

-- Export for use as API
if not ... then
    main()
else
    return {
        sha256 = sha256,
        mine = mine,
        stats = stats
    }
end