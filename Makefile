CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = -pthread -ljsoncpp

# Archivos fuente
SOURCES = brkga_mdvrp.cpp benchmark_search.cpp
HEADERS = brkga_mdvrp.h

# Nombre del ejecutable
TARGET = benchmark_search

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean