# Bckn CPU Miner Makefile
# Optimized for Ubuntu/Debian systems

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mtune=native -flto -fomit-frame-pointer -funroll-loops -fprefetch-loop-arrays
LDFLAGS = -lcurl -ljsoncpp -lcrypto -lpthread -flto

# Executable name
TARGET = bckn-miner-cpu

# Source files
SOURCES = bckn-miner-cpu.cpp

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	strip $(TARGET)

# Debug build
debug: CXXFLAGS = -std=c++17 -g -O0 -DDEBUG
debug: $(TARGET)

# Install dependencies (requires sudo)
install-deps:
	sudo apt-get update
	sudo apt-get install -y build-essential libcurl4-openssl-dev libjsoncpp-dev libssl-dev

# Clean build files
clean:
	rm -f $(TARGET)

# Run the miner (requires private key as argument)
run: $(TARGET)
	@echo "Usage: ./$(TARGET) <private_key>"
	@echo "Example: ./$(TARGET) your_private_key_here"

# Install the miner system-wide
install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/
	@echo "Installed to /usr/local/bin/$(TARGET)"

# Uninstall
uninstall:
	sudo rm -f /usr/local/bin/$(TARGET)

.PHONY: all debug install-deps clean run install uninstall