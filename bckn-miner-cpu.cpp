/*
 * Bckn CPU Miner - High Performance C++ Implementation
 * Optimized for RunPod Ubuntu CPU instances
 */

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <curl/curl.h>
#ifdef __has_include
  #if __has_include(<jsoncpp/json/json.h>)
    #include <jsoncpp/json/json.h>
  #else
    #include <json/json.h>
  #endif
#else
  #include <json/json.h>
#endif
#include <openssl/sha.h>

const std::string BCKN_NODE = "https://bckn.dev";
const int BATCH_SIZE = 1000000;

// Global atomic variables for thread coordination
std::atomic<bool> found(false);
std::atomic<uint64_t> global_counter(0);
std::atomic<uint64_t> found_nonce(0);
std::string found_hash;
std::atomic<bool> should_exit(false);

// Signal handler for clean shutdown
void signal_handler(int signum) {
    std::cout << "\n\nShutting down gracefully..." << std::endl;
    should_exit = true;
    found = true; // Stop mining threads
}

// Curl write callback
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* out) {
    size_t total = size * nmemb;
    out->append((char*)contents, total);
    return total;
}

// HTTP GET request
std::string http_get(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string response;
    
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        curl_easy_cleanup(curl);
    }
    return response;
}

// HTTP POST request
std::string http_post(const std::string& url, const std::string& json_data) {
    CURL* curl = curl_easy_init();
    std::string response;
    
    if (curl) {
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    return response;
}

// Convert bytes to hex string
std::string bytes_to_hex(const unsigned char* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    return ss.str();
}

// Fast hex to integer conversion for first 12 chars
inline uint64_t hex12_to_int(const unsigned char* hash) {
    uint64_t result = 0;
    for (int i = 0; i < 6; ++i) {
        result = (result << 8) | hash[i];
    }
    return result;
}

// Mining worker thread
void mine_worker(int thread_id, const std::string& prefix, uint64_t work, uint64_t start_nonce) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    
    uint64_t nonce = start_nonce + (thread_id * 1000000000ULL);
    uint64_t local_counter = 0;
    
    // Pre-allocate buffer for message
    std::string message_base = prefix;
    size_t prefix_len = prefix.length();
    char nonce_buffer[32];
    
    while (!found.load()) {
        // Process batch
        for (int i = 0; i < BATCH_SIZE && !found.load(); ++i) {
            // Convert nonce to string
            int nonce_len = snprintf(nonce_buffer, sizeof(nonce_buffer), "%llu", nonce);
            
            // Calculate SHA256
            SHA256_Init(&sha256);
            SHA256_Update(&sha256, prefix.c_str(), prefix_len);
            SHA256_Update(&sha256, nonce_buffer, nonce_len);
            SHA256_Final(hash, &sha256);
            
            // Quick check using first 6 bytes (48 bits)
            uint64_t hash_value = hex12_to_int(hash);
            
            if (hash_value <= work) {
                // Double check with full hex conversion
                std::string hash_hex = bytes_to_hex(hash, SHA256_DIGEST_LENGTH);
                uint64_t full_value = std::stoull(hash_hex.substr(0, 12), nullptr, 16);
                
                if (full_value <= work) {
                    bool expected = false;
                    if (found.compare_exchange_strong(expected, true)) {
                        found_nonce = nonce;
                        found_hash = hash_hex;
                    }
                    return;
                }
            }
            
            nonce++;
            local_counter++;
        }
        
        // Update global counter periodically
        if (local_counter >= 10000000) {
            global_counter += local_counter;
            local_counter = 0;
        }
    }
}

// Stats printer thread
void stats_printer(std::chrono::steady_clock::time_point start_time) {
    while (!found.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        
        if (elapsed > 0) {
            double hashrate = static_cast<double>(global_counter.load()) / elapsed;
            std::cout << "\r" << std::fixed << std::setprecision(2) 
                      << hashrate / 1000000.0 << " MH/s | "
                      << global_counter.load() / 1000000000.0 << "B hashes | "
                      << elapsed << "s" << std::flush;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <private_key> [--once]" << std::endl;
        std::cerr << "  --once: Mine single block then exit (default: continuous)" << std::endl;
        return 1;
    }
    
    bool continuous_mode = true;
    if (argc > 2 && std::string(argv[2]) == "--once") {
        continuous_mode = false;
    }
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Set up signal handler for clean shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    std::cout << "=== Bckn CPU Miner (C++ Edition) ===" << std::endl;
    if (continuous_mode) {
        std::cout << "Mode: Continuous (Ctrl+C to stop)" << std::endl;
    }
    
    // Login
    Json::Value login_data;
    login_data["privatekey"] = argv[1];
    Json::FastWriter writer;
    std::string login_json = writer.write(login_data);
    
    std::string login_response = http_post(BCKN_NODE + "/login", login_json);
    
    Json::Reader reader;
    Json::Value login_result;
    if (!reader.parse(login_response, login_result)) {
        std::cerr << "Failed to parse login response" << std::endl;
        return 1;
    }
    
    std::string address = login_result["address"].asString();
    std::cout << "Address: " << address << std::endl;
    
    // Statistics tracking
    int blocks_found = 0;
    auto session_start = std::chrono::steady_clock::now();
    
    // Main mining loop
    while (!should_exit && (continuous_mode || blocks_found == 0)) {
    
    // Get work
    std::string work_response = http_get(BCKN_NODE + "/work");
    Json::Value work_result;
    if (!reader.parse(work_response, work_result)) {
        std::cerr << "Failed to parse work response" << std::endl;
        return 1;
    }
    
    uint64_t work = work_result["work"].asUInt64();
    std::cout << "Work: " << work << std::endl;
    
    // Get last block
    std::string blocks_response = http_get(BCKN_NODE + "/blocks/last");
    Json::Value blocks_result;
    if (!reader.parse(blocks_response, blocks_result)) {
        std::cerr << "Failed to parse blocks response" << std::endl;
        return 1;
    }
    
    std::string last_hash = "000000000000";
    if (blocks_result.isMember("block") && blocks_result["block"].isMember("hash")) {
        last_hash = blocks_result["block"]["hash"].asString().substr(0, 12);
    }
    std::cout << "Last hash: " << last_hash << std::endl;
    
    // Prepare for mining
    std::string prefix = address + last_hash;
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::cout << "Starting " << num_threads << " mining threads..." << std::endl << std::endl;
    
    // Reset counters
    found = false;
    global_counter = 0;
    
    // Random starting nonce
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, 1ULL << 32);
    uint64_t start_nonce = dis(gen);
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Start stats printer
    std::thread stats_thread(stats_printer, start_time);
    
    // Start mining threads
    std::vector<std::thread> miners;
    for (unsigned int i = 0; i < num_threads; ++i) {
        miners.emplace_back(mine_worker, i, prefix, work, start_nonce);
    }
    
    // Wait for solution
    for (auto& t : miners) {
        t.join();
    }
    
    // Stop stats printer
    found = true;
    stats_thread.join();
    
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    
    std::cout << "\n\n💎 Found valid nonce! " << found_nonce.load() << std::endl;
    std::cout << "Hash: " << found_hash << std::endl;
    std::cout << "Value: " << std::stoull(found_hash.substr(0, 12), nullptr, 16) << " <= " << work << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(1) << elapsed << "s at " 
              << (global_counter.load() / elapsed / 1000000.0) << " MH/s" << std::endl;
    
    // Submit block
    Json::Value submit_data;
    submit_data["address"] = address;
    submit_data["nonce"] = std::to_string(found_nonce.load());
    std::string submit_json = writer.write(submit_data);
    
    std::string submit_response = http_post(BCKN_NODE + "/submit", submit_json);
    Json::Value submit_result;
    if (reader.parse(submit_response, submit_result)) {
        if (submit_result["success"].asBool()) {
            blocks_found++;
            std::cout << "✓ Block submitted successfully!" << std::endl;
            std::cout << "Reward: 25 BCN | Total blocks: " << blocks_found << " | Total earned: " << (blocks_found * 25) << " BCN" << std::endl;
        } else {
            std::cout << "✗ Submission failed: " << submit_response << std::endl;
        }
    }
    
    if (continuous_mode && !should_exit) {
        std::cout << "\nStarting new round in 3 seconds...\n" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
    
    } // end while loop
    
    // Show session statistics
    if (blocks_found > 0) {
        auto session_end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(session_end - session_start).count();
        std::cout << "\n=== Session Summary ===" << std::endl;
        std::cout << "Total blocks found: " << blocks_found << std::endl;
        std::cout << "Total BCN earned: " << (blocks_found * 25) << std::endl;
        std::cout << "Session duration: " << duration << " seconds" << std::endl;
    }
    
    curl_global_cleanup();
    return 0;
}