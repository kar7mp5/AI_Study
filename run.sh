#!/bin/bash

# Create 'build' directory if it doesn't exist
if [ ! -d "build" ]; then
  echo "'build' directory not found. Creating it..."
  mkdir build
fi

# Change to the build directory
cd build || { echo "Failed to enter 'build' directory. Exiting."; exit 1; }

# Check if 'build' file exists
if [ ! -f "build" ]; then
  echo "'build' file not found. Cleaning up and rebuilding..."

  # Delete unusual folders and files
  rm -rf CMakeFiles
  rm -f cmake_install.cmake CMakeCache.txt main Makefile

  # Check if CMakeLists.txt exists in the parent directory
  if [ ! -f "../CMakeLists.txt" ]; then
    echo "CMakeLists.txt not found in the parent directory. Exiting."
    exit 1
  fi

  # Build and run the code
  cmake ..
  make

  # Check if the executable 'main' was successfully built
  if [ -f "./main" ]; then
    ./main
  else
    echo "Executable 'main' not found. Build might have failed."
    exit 1
  fi
else
  echo "'build' file exists. Checking contents..."

  # Check if 'build' file is empty
  if [ ! -s "build" ]; then
    echo "'build' file is empty. Rebuilding..."

    # Build and run the code
    cmake ..
    make

    # Check if the executable 'main' was successfully built
    if [ -f "./main" ]; then
      ./main
    else
      echo "Executable 'main' not found. Build might have failed."
      exit 1
    fi
  else
    echo "'build' file is not empty. Nothing to do."
  fi
fi