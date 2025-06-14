#include <ros/ros.h> 
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h> // For pcl::fromROSMsg
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h> // For saving PCD files


#include <spdlog/spdlog.h>        // For logging


#include <tf/transform_datatypes.h>   // For tf::Matrix3x3, tf::Vector3, tf::Transform, tf::Quaternion
#include <tf_conversions/tf_eigen.h>  // Essential for tf::transformEigenToTF

// Crucial for std::filesystem, smart pointers, and file I/O
#include <filesystem>
#include <memory> // For std::unique_ptr
#include <fstream> // For file output (ofstream)
#include <iomanip> // For std::fixed and std::setprecision

// Include our ICP class
#include "icp.h"

// Define PointCloud types for convenience
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ_NoColor;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB_Color;

// Use std::unique_ptr for automatic memory management
std::unique_ptr<ICPRegistration> g_icp_registrator_ptr;

// Global variable to store the previous point cloud for ICP (current aligns to previous)
PointCloudXYZ_NoColor::Ptr g_previous_cloud_icp(new PointCloudXYZ_NoColor);
// Global variable to store the previous point cloud for display (colorized)
PointCloudXYZRGB_Color::Ptr g_previous_cloud_rgb_display(new PointCloudXYZRGB_Color);

bool g_first_cloud_processed = false; // Flag to indicate if at least one cloud has been processed

// Counters for saved PCD files
int g_cloud_sequence_number = 0; // Sequential counter for all processed clouds

// Output directories and pose file path
const std::string RESULTS_BASE_DIR = "/home/haochen/code/slam_trick/icp/results";
const std::string TARGET_OUTPUT_DIR = RESULTS_BASE_DIR + "/output_target_clouds";
const std::string ALIGNED_OUTPUT_DIR = RESULTS_BASE_DIR + "/output_aligned_clouds";
const std::string POSE_OUTPUT_FILE = RESULTS_BASE_DIR + "/odometry_poses.txt";

std::ofstream g_pose_file; // Global file stream for writing poses

// Global variable to accumulate transformations for odometry
Eigen::Matrix4f g_accumulated_transform = Eigen::Matrix4f::Identity();


void processPointCloud(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
    spdlog::info("Processing PointCloud2 message (sequence: {}).", g_cloud_sequence_number + 1);

    // Convert incoming message to PointXYZRGB_Color
    PointCloudXYZRGB_Color::Ptr current_cloud_rgb(new PointCloudXYZRGB_Color);
    pcl::fromROSMsg(*cloud_msg, *current_cloud_rgb);

    if (current_cloud_rgb->empty())
    {
        spdlog::warn("Received empty point cloud.");
        return;
    }

    // Convert PointXYZRGB_Color to PointCloudXYZ_NoColor for ICP processing
    PointCloudXYZ_NoColor::Ptr current_cloud_xyz_icp(new PointCloudXYZ_NoColor);
    pcl::copyPointCloud(*current_cloud_rgb, *current_cloud_xyz_icp);

    // Lazy initialization of ICP registrator using std::make_unique
    if (!g_icp_registrator_ptr)
    {
        g_icp_registrator_ptr = std::make_unique<ICPRegistration>();
        spdlog::info("Initialized ICP registrator (using std::unique_ptr).");
        
        
    }

    if (!g_first_cloud_processed)
    {
        // For the very first cloud, it becomes the 'previous' frame for the next iteration.
        // There's no 'previous' to align to, so we just store it.
        *g_previous_cloud_icp = *current_cloud_xyz_icp;
        *g_previous_cloud_rgb_display = *current_cloud_rgb; // Keep a colorized version for saving

        // Color the first cloud (green) and save it as the initial 'target'
        for (auto& point : g_previous_cloud_rgb_display->points)
        {
            point.r = 0;
            point.g = 255;
            point.b = 0;
        }

        g_cloud_sequence_number++;
        std::string initial_target_filename = TARGET_OUTPUT_DIR + "/cloud_" +
                                               std::to_string(g_cloud_sequence_number) + ".pcd";
        pcl::io::savePCDFileASCII(initial_target_filename, *g_previous_cloud_rgb_display);
        spdlog::info("First point cloud stored as initial frame and saved to: {}", initial_target_filename);

        // Record the initial pose (identity)
        // Format: timestamp x y z qx qy qz qw (or sequence_number x y z qx qy qz qw)
        // For now, using sequence_number as timestamp placeholder
        if (g_pose_file.is_open()) {
            g_pose_file << std::fixed << std::setprecision(6)
                        << g_cloud_sequence_number << " "
                        << g_accumulated_transform(0, 3) << " "
                        << g_accumulated_transform(1, 3) << " "
                        << g_accumulated_transform(2, 3) << " "
                        << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl; // Identity quaternion
        }

        g_first_cloud_processed = true;
        return;
    }

    // Apply color to the current cloud for saving (RED before alignment)
    for (auto& point : current_cloud_rgb->points)
    {
        point.r = 255; // Set red channel to maximum
        point.g = 0;   // Set green channel to minimum
        point.b = 0;   // Set blue channel to minimum
    }

    // Perform ICP: Align current cloud (source) to the previous cloud (target)
    Eigen::Matrix4f transformation_current_to_previous =
        g_icp_registrator_ptr->align(current_cloud_xyz_icp, g_previous_cloud_icp);

    // Accumulate the transformation for odometry
    g_accumulated_transform = g_accumulated_transform * transformation_current_to_previous;

    // Apply the transformation to the current cloud (the RGB version for saving)
    // This transform aligns the current_cloud into the previous_cloud's coordinate system
    PointCloudXYZRGB_Color::Ptr aligned_current_cloud_rgb(new PointCloudXYZRGB_Color);
    pcl::transformPointCloud(*current_cloud_rgb, *aligned_current_cloud_rgb, transformation_current_to_previous);

    // Set aligned cloud color to BLUE for saving
    for (auto& point : aligned_current_cloud_rgb->points)
    {
        point.r = 0;
        point.g = 0;
        point.b = 255;
    }

    // Save the aligned current cloud (aligned to the previous frame)
    g_cloud_sequence_number++;
    std::string aligned_filename = ALIGNED_OUTPUT_DIR + "/aligned_" +
                                   std::to_string(g_cloud_sequence_number) + ".pcd";
    pcl::io::savePCDFileASCII(aligned_filename, *aligned_current_cloud_rgb);
    spdlog::info("Aligned cloud (sequence {}) saved to: {}", g_cloud_sequence_number, aligned_filename);

    // Save the previous cloud (which now acts as a 'target' for comparison in the aligned folder)
    // We can save the previous cloud (which was target for this alignment) with a different color.
    for (auto& point : g_previous_cloud_rgb_display->points) {
        point.r = 0; point.g = 255; point.b = 0;
    }
    std::string previous_target_filename = ALIGNED_OUTPUT_DIR + "/target_prev_" +
                                           std::to_string(g_cloud_sequence_number - 1) + ".pcd";
    pcl::io::savePCDFileASCII(previous_target_filename, *g_previous_cloud_rgb_display);
    spdlog::info("Previous target cloud (sequence {}) saved to: {}", g_cloud_sequence_number - 1, previous_target_filename);


    // Update g_previous_cloud_icp for the next iteration (current becomes previous)
    *g_previous_cloud_icp = *current_cloud_xyz_icp;
    *g_previous_cloud_rgb_display = *current_cloud_rgb; // Store the *untransformed* current cloud as the new previous for display.

    // Log the relative transformation (current to previous)
    spdlog::info("ICP Relative Transformation Matrix (current to previous translation): x {}, y {}, z {}",
                 transformation_current_to_previous(0, 3), transformation_current_to_previous(1, 3), transformation_current_to_previous(2, 3));

    Eigen::Affine3d relative_eigen_transform_double = Eigen::Affine3d::Identity();
    relative_eigen_transform_double.matrix() = transformation_current_to_previous.cast<double>();
    tf::Transform relative_tf_transform;
    tf::transformEigenToTF(relative_eigen_transform_double, relative_tf_transform);
    spdlog::info("Relative Rotation quaternion: x {}, y {}, z {}, w {}",
                 relative_tf_transform.getRotation().x(),
                 relative_tf_transform.getRotation().y(),
                 relative_tf_transform.getRotation().z(),
                 relative_tf_transform.getRotation().w());

    // Log the accumulated (odometry) transformation
    spdlog::info("Accumulated Odometry Transformation Matrix (translation): x {}, y {}, z {}",
                 g_accumulated_transform(0, 3), g_accumulated_transform(1, 3), g_accumulated_transform(2, 3));

    Eigen::Affine3d accumulated_eigen_transform_double = Eigen::Affine3d::Identity();
    accumulated_eigen_transform_double.matrix() = g_accumulated_transform.cast<double>();
    tf::Transform accumulated_tf_transform;
    tf::transformEigenToTF(accumulated_eigen_transform_double, accumulated_tf_transform);
    spdlog::info("Accumulated Odometry Rotation quaternion: x {}, y {}, z {}, w {}",
                 accumulated_tf_transform.getRotation().x(),
                 accumulated_tf_transform.getRotation().y(),
                 accumulated_tf_transform.getRotation().z(),
                 accumulated_tf_transform.getRotation().w());

    // Write the accumulated pose to the file
    if (g_pose_file.is_open()) {
        g_pose_file << std::fixed << std::setprecision(6) // Set precision for float output
                    << g_cloud_sequence_number << " "     // Frame sequence number (timestamp placeholder)
                    << accumulated_tf_transform.getOrigin().x() << " "
                    << accumulated_tf_transform.getOrigin().y() << " "
                    << accumulated_tf_transform.getOrigin().z() << " "
                    << accumulated_tf_transform.getRotation().x() << " "
                    << accumulated_tf_transform.getRotation().y() << " "
                    << accumulated_tf_transform.getRotation().z() << " "
                    << accumulated_tf_transform.getRotation().w() << std::endl;
    }

    spdlog::info("ICP complete. Fitness score: {}, Converged: {}",
                 g_icp_registrator_ptr->getFitnessScore(),
                 g_icp_registrator_ptr->hasConverged() ? "true" : "false");
}

int main(int argc, char** argv)
{
    // Initialize spdlog before ROS (optional, but good practice)
    spdlog::set_level(spdlog::level::info); // Set global log level
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); // Simplified pattern

    ros::init(argc, argv, "rosbag_icp_pcd_saver");
    ros::NodeHandle nh; // Still needed for rosbag interaction

    if (argc < 2)
    {
        spdlog::error("Usage: rosrun your_package_name rosbag_icp_pcd_saver <path_to_rosbag_file>");
        return 1;
    }

    std::string bag_path = argv[1];

    // Create the base results directory and subdirectories
    try
    {
        std::filesystem::create_directories(RESULTS_BASE_DIR);
        spdlog::info("Created base results directory: {}", RESULTS_BASE_DIR);
        std::filesystem::create_directories(TARGET_OUTPUT_DIR);
        spdlog::info("Created target clouds directory: {}", TARGET_OUTPUT_DIR);
        std::filesystem::create_directories(ALIGNED_OUTPUT_DIR);
        spdlog::info("Created aligned clouds directory: {}", ALIGNED_OUTPUT_DIR);

        // Open the pose output file
        g_pose_file.open(POSE_OUTPUT_FILE);
        if (!g_pose_file.is_open()) {
            spdlog::error("Failed to open pose output file: {}", POSE_OUTPUT_FILE);
            return 1;
        }
        spdlog::info("Opened pose output file: {}", POSE_OUTPUT_FILE);
        // Write header to pose file if desired
        g_pose_file << "# sequence_number x y z qx qy qz qw" << std::endl;
    }
    catch (const std::filesystem::filesystem_error& e)
    {
        spdlog::error("Filesystem error creating directories or opening pose file: {}", e.what());
        return 1;
    }
    catch (const std::exception& e)
    {
        spdlog::error("General error during setup: {}", e.what());
        return 1;
    }


    rosbag::Bag bag;
    try
    {
        bag.open(bag_path, rosbag::bagmode::Read);
    }
    catch (rosbag::BagException& e)
    {
        spdlog::error("Error opening bag file: {}", e.what());
        return 1;
    }
    std::vector<std::string> topics;
    topics.push_back("/velodyne_points"); // **CHANGE THIS TO YOUR POINT CLOUD TOPIC** (e.g., from `rosbag info`)
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Process messages in the bag
    for (rosbag::MessageInstance const m : view)
    {
        if (ros::ok())
        {
            sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (cloud_msg != nullptr)
            {
                processPointCloud(cloud_msg);
            }
        }
        else
        {
            break; // Exit if ROS is shutting down
        }
    }

    bag.close();
    spdlog::info("Finished processing rosbag and saving PCDs.");

    // Close the pose file
    if (g_pose_file.is_open()) {
        g_pose_file.close();
        spdlog::info("Closed pose output file.");
    }

    // Smart pointer takes care of deallocation, no need for `delete g_icp_registrator_ptr;`

    return 0;
}