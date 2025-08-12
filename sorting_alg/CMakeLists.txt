cmake_minimum_required(VERSION 3.10)

# Set the project name
project(SortingAlgorithms VERSION 1.0)

# Specify the C++ standard to use
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the source files for the sorting library
# We can create a static library for the sorting functions
# This is good practice for separating reusable code
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(sorting_lib
    src/sort.cpp
)

# Tell CMake where to find the header files
# The "include" directory is a public include directory
target_include_directories(sorting_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define the executable
add_executable(sorting_app src/main.cpp)

# Link the executable with the sorting library
# The executable "sorting_app" depends on the "sorting_lib"
target_link_libraries(sorting_app PRIVATE sorting_lib)