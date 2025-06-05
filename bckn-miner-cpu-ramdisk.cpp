/*
 * Bckn CPU Miner - RAM-Optimized Edition
 * Uses memory-mapped tables and huge pages for maximum performance
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
#include <sys/mman.h>
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
const int BATCH_SIZE = 10000000;  // 10x larger batches
const size_t CACHE_SIZE = 1ULL << 30; // 1GB lookup table

// Global atomic variables for thread coordination
std::atomic<bool> found(false);
std::atomic<uint64_t> global_counter(0);
std::atomic<uint64_t> found_nonce(0);
std::string found_hash;

// Precomputed nonce string cache
struct NonceCache {
    char* buffer;
    size_t* offsets;
    size_t* lengths;
    size_t count;
    
    NonceCache(size_t max_nonce) : count(max_nonce) {
        // Allocate with huge pages if available
        size_t buffer_size = max_nonce * 20; // Max 20 chars per nonce
        buffer = (char*)mmap(nullptr, buffer_size, 
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                           -1, 0);
        
        if (buffer == MAP_FAILED) {
            // Fallback to regular pages
            buffer = (char*)mmap(nullptr, buffer_size, 
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS,
                               -1, 0);
        }
        
        offsets = new size_t[max_nonce];
        lengths = new size_t[max_nonce];
        
        // Pre-generate all nonce strings
        size_t offset = 0;
        for (size_t i = 0; i < max_nonce; ++i) {
            offsets[i] = offset;
            lengths[i] = sprintf(buffer + offset, "%llu", (unsigned long long)i);
            offset += lengths[i];
        }
        
        // Advise kernel about access pattern
        madvise(buffer, buffer_size, MADV_SEQUENTIAL);
    }
    
    ~NonceCache() {
        // Cleanup handled by OS
    }
    
    inline void get_nonce(size_t n, const char** str, size_t* len) {
        *str = buffer + offsets[n];
        *len = lengths[n];
    }
};

// Curl callbacks
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* out) {
    size_t total = size * nmemb;
    out->append((char*)contents, total);
    return total;
}

// HTTP functions (same as before)
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

// Mining worker thread with cache
void mine_worker_cached(int thread_id, const std::string& prefix, uint64_t work, 
                       uint64_t start_nonce, NonceCache* cache) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256_base, sha256;
    
    // Pre-initialize with prefix
    SHA256_Init(&sha256_base);
    SHA256_Update(&sha256_base, prefix.c_str(), prefix.length());
    
    uint64_t nonce = start_nonce + (thread_id * 100000000ULL);
    uint64_t local_counter = 0;
    
    // Process in huge batches
    while (!found.load()) {
        uint64_t batch_end = std::min(nonce + BATCH_SIZE, cache->count);
        
        for (; nonce < batch_end && !found.load(); ++nonce) {
            // Copy pre-initialized context
            memcpy(&sha256, &sha256_base, sizeof(SHA256_CTX));
            
            // Get cached nonce string
            const char* nonce_str;
            size_t nonce_len;
            cache->get_nonce(nonce % cache->count, &nonce_str, &nonce_len);
            
            // Complete hash
            SHA256_Update(&sha256, nonce_str, nonce_len);
            SHA256_Final(hash, &sha256);
            
            // Quick check using first 6 bytes
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
            
            local_counter++;
        }
        
        // Update global counter
        if (local_counter >= 100000000) {
            global_counter += local_counter;
            local_counter = 0;
        }
        
        // Wrap around if needed
        if (nonce >= cache->count) {
            nonce = thread_id * 100000000ULL;
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
                      << elapsed << "s | RAM-Optimized" << std::flush;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <private_key>" << std::endl;
        return 1;
    }
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    std::cout << "=== Bckn CPU Miner (RAM-Optimized Edition) ===" << std::endl;
    std::cout << "Allocating RAM cache..." << std::endl;
    
    // Create nonce cache (uses ~2GB RAM for 100M nonces)
    NonceCache* cache = new NonceCache(100000000);
    std::cout << "Cache ready: 100M pre-computed nonces" << std::endl;
    
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
    std::uniform_int_distribution<uint64_t> dis(0, 50000000);
    uint64_t start_nonce = dis(gen);
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Start stats printer
    std::thread stats_thread(stats_printer, start_time);
    
    // Start mining threads
    std::vector<std::thread> miners;
    for (unsigned int i = 0; i < num_threads; ++i) {
        miners.emplace_back(mine_worker_cached, i, prefix, work, start_nonce, cache);
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
            std::cout << "✓ Block submitted successfully!" << std::endl;
            std::cout << "Reward: 25 BCN" << std::endl;
        } else {
            std::cout << "✗ Submission failed: " << submit_response << std::endl;
        }
    }
    
    delete cache;
    curl_global_cleanup();
    return 0;
}