cmake_minimum_required(VERSION 3.10) # Using 3.10 for better C++17 support and modern CMake practices
project(icp CXX) # Specify CXX language

set(CMAKE_CXX_STANDARD 17) # Explicitly set C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) 

# --- Find Dependencies ---
find_package(rosbag REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf REQUIRED) # For core TF library
find_package(tf_conversions REQUIRED) # For tf_eigen.h and tf::transformEigenToTF

# Find PCL
find_package(PCL REQUIRED COMPONENTS common io registration)

# Find Eigen
find_package(Eigen3 REQUIRED CONFIG) # Use CONFIG for modern Eigen installations

# Find spdlog
find_package(spdlog REQUIRED CONFIG) # Use CONFIG for spdlog built with CMake

# Find Boost for filesystem exception (often pulled by ROS/PCL, but good to be explicit)
# We only need it if the boost::filesystem::filesystem_error catch is used.
find_package(Boost 1.58 COMPONENTS system filesystem REQUIRED)


# --- Include Directories ---

include_directories(
  ${CMAKE_SOURCE_DIR}/include 
  ${ros_comm_INCLUDE_DIRS}
  ${rosbag_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${tf_INCLUDE_DIRS}
  ${tf_conversions_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${spdlog_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)


# --- Libraries ---

add_library(icp_lib
  src/icp.cpp
)

target_link_libraries(icp_lib
  # PCL and its dependencies
  ${PCL_LIBRARIES}

  
  Eigen3::Eigen


  spdlog::spdlog
)

add_executable(rosbag_icp_pcd_saver
  src/main.cpp
)

target_link_libraries(rosbag_icp_pcd_saver
  icp_lib 
 
  ${ros_comm_LIBRARIES}
  ${rosbag_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${tf_LIBRARIES}
  ${tf_conversions_LIBRARIES} # Link tf_conversions library

  # PCL and its dependencies
  ${PCL_LIBRARIES}
  
  # Eigen
  Eigen3::Eigen

  # spdlog
  spdlog::spdlog

  # Boost (for filesystem, if used explicitly in catch blocks)
  ${Boost_LIBRARIES}
)


#second script

add_executable(icp_manual src/icp_bt.cpp)
target_link_libraries(icp_manual
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    spdlog::spdlog
)
