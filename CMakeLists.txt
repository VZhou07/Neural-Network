cmake_minimum_required(VERSION 3.15)

project(NumberClassifier VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main src/main.cpp src/mnist_loader.cpp)
target_include_directories(main PRIVATE include)
