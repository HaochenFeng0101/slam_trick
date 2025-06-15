cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(ndt_registration CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PCL
find_package(PCL 1.10 REQUIRED COMPONENTS common io registration filters visualization)
if (PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else()
    message(FATAL_ERROR "PCL not found. Please install PCL or set PCL_ROOT.")
endif (PCL_FOUND)

# Find Eigen3
find_package(Eigen3 CONFIG REQUIRED)
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

find_package(Boost REQUIRED COMPONENTS system filesystem)
# Add  source file
add_executable(ndt_registration ndt_registration.cpp)

# Link libraries
target_link_libraries(ndt_registration
    ${PCL_LIBRARIES}
    ${spdlog_LIBRARIES}
)



#####
add_executable(ndt_manual_calculation ndt_calculation.cpp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
)
# Link libraries
target_link_libraries(ndt_manual_calculation
    ${spdlog_LIBRARIES}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
 

)
