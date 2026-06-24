cmake_minimum_required(VERSION 3.15)

project(NumberClassifier VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
)

FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/archive/refs/tags/4.5.0.tar.gz
)

FetchContent_MakeAvailable(raylib)

FetchContent_MakeAvailable(Eigen3)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main src/main.cpp src/mnist_loader.cpp src/neural_net.cpp)
target_include_directories(main PRIVATE include)
target_link_libraries(main PRIVATE Eigen3::Eigen)
target_link_libraries(main PRIVATE raylib)
target_link_libraries(main PRIVATE Threads::Threads)