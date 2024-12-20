cmake_minimum_required(VERSION 3.10)

project(main VERSION 1.0 LANGUAGES C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES
    src/ConvolutionNeuralNetwork.c
)

add_executable(main ${SOURCES})

include_directories(include)
