cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(lidar_feature_extractor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PCL
find_package(PCL 1.10 REQUIRED COMPONENTS common io kdtree) # PCL 1.10 is common on Ubuntu 20.04/22.04
if (PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else()
    message(FATAL_ERROR "PCL not found. Please install PCL or set PCL_ROOT.")
endif (PCL_FOUND)

# Find Eigen3
find_package(Eigen3 CONFIG REQUIRED) # Uses CMake's Eigen3 config module
if (Eigen3_FOUND)
    include_directories(${Eigen3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 or set Eigen3_DIR.")
endif (Eigen3_FOUND)

# Find spdlog


find_package(spdlog CONFIG REQUIRED)
if (spdlog_FOUND)
    include_directories(${spdlog_INCLUDE_DIR})
else()
    message(FATAL_ERROR "spdlog not found. Please install spdlog or set spdlog_DIR.")
endif (spdlog_FOUND)

# Add your source files
set(SOURCE_FILES
    src/feature_extractor.cpp
    src/main.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
)

# Add the executable
add_executable(lidar_feature_extractor ${SOURCE_FILES})

# Link libraries
target_link_libraries(lidar_feature_extractor
    ${PCL_LIBRARIES}
    ${spdlog_LIBRARIES}
)