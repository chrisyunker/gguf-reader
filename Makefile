CXX      := c++
CXXFLAGS := -std=c++17 -O2 -Wall
TARGET   := build/gguf-reader
SRC      := src/main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) | build
	$(CXX) $(CXXFLAGS) -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build
