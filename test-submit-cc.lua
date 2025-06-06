-- Test nonce submission for ComputerCraft
-- This tests if CC can successfully submit a known valid nonce

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

-- Main test function
print("=== Bckn Nonce Submission Test ===")
print("")

-- Get inputs
print("Enter your mining address:")
local address = read()

print("Enter the valid nonce to submit:")
local nonce = read()

print("Enter your private key (for verification):")
local privateKey = read("*")

print("\nVerifying address...")
local loginResp = httpRequest("POST", "/login", {privatekey = privateKey})
if not loginResp or not loginResp.address then
    print("Login failed!")
    return
end

if loginResp.address ~= address then
    print("Address mismatch! Login returned: " .. loginResp.address)
    print("You entered: " .. address)
    return
end

print("Address verified!")

-- Get current work to display
local workResp = httpRequest("GET", "/work")
local lastBlockResp = httpRequest("GET", "/blocks/last")

if workResp and lastBlockResp then
    local work = workResp.work
    local lastHash = lastBlockResp.block and lastBlockResp.block.hash:sub(1, 12) or "000000000000"
    
    print("\nCurrent network state:")
    print("Work: " .. work)
    print("Last hash: " .. lastHash)
end

print("\nSubmitting nonce...")
print("Address: " .. address)
print("Nonce: " .. nonce)

-- Submit the nonce
local submitResp = httpRequest("POST", "/submit", {
    address = address,
    nonce = tostring(nonce)
})

print("\nSubmission response:")
print(textutils.serialize(submitResp))

if submitResp and submitResp.success then
    print("\n*** SUCCESS! Block submitted! ***")
    print("You should have received 25 BCN!")
else
    print("\n*** Submission failed ***")
    if submitResp and submitResp.error then
        print("Error: " .. submitResp.error)
    end
end