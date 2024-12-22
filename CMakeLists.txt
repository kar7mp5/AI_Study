cmake_minimum_required(VERSION 3.10)
project(MatrixLibrary)

# Set the C standard version
set(CMAKE_C_STANDARD 99)

# Include the include directory for header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create the main executable and link it with the matrix library
add_executable(main 
    src/ConvolutionNeuralNetwork.c
    src/matrix/matmul.c
)

# Link the library to the main executable
target_link_libraries(main)

# Add a custom command to run the program after building
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/main
    COMMENT "Running the program after build"
)