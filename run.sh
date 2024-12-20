#! /bin/bash
cd build

# Delete unusal folder and files
rm -rf CMakeFiles
rm cmake_install.cmake CMakeCache.txt main Makefile

# Build and run the code
cmake ..
make
./main