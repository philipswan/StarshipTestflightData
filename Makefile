# Makefile for extractTelemetry project

# Compiler and flags
CXX      = g++
CXXFLAGS = -std=c++20 -O2 -Wall

# Use pkg-config to get the include and library flags for OpenCV and JsonCpp
OPENCV   = `pkg-config --cflags --libs opencv4`
JSONCPP  = `pkg-config --cflags --libs jsoncpp`

# Tesseract flags: include directory and linking flags
TESSERACT = -I/usr/include/tesseract -L/usr/lib/x86_64-linux-gnu -ltesseract -Wno-deprecated-enum-enum-conversion

# List your source files here
SRC      = extractTelemetry.cpp OrientationDetector.cpp
OBJ      = $(SRC:.cpp=.o)
TARGET   = extractTelemetry

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(OPENCV) $(JSONCPP) $(TESSERACT)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< $(OPENCV) $(JSONCPP) $(TESSERACT)

# Clean up build artifacts
clean:
	rm -f $(OBJ) $(TARGET)
