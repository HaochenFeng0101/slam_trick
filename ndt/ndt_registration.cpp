#include <iostream>
#include <string>
#include <chrono> // For timing the process

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h> // For downsampling
#include <pcl/visualization/pcl_visualizer.h>   // For visualization (optional)
#include <spdlog/spdlog.h>                      // For logging
#include <sstream> // For formatted output
// Define the point type to use (e.g., PointXYZ for simple 3D points)
using PointT = pcl::PointXYZ;

// Function to visualize point clouds
void visualizeClouds(const pcl::PointCloud<PointT>::Ptr& cloud_source,
                     const pcl::PointCloud<PointT>::Ptr& cloud_target,
                     const pcl::PointCloud<PointT>::Ptr& cloud_aligned)
{
    spdlog::info("Starting visualization. Close the viewer to continue.");

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("NDT Registration Viewer"));
    viewer->setBackgroundColor(0, 0, 0); // Black background

    // Add source cloud (red)
    pcl::visualization::PointCloudColorHandlerCustom<PointT> source_color(cloud_source, 255, 0, 0);
    viewer->addPointCloud<PointT>(cloud_source, source_color, "source cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source cloud");

    // Add target cloud (green)
    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_color(cloud_target, 0, 255, 0);
    viewer->addPointCloud<PointT>(cloud_target, target_color, "target cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");

    // Add aligned cloud (blue)
    pcl::visualization::PointCloudColorHandlerCustom<PointT> aligned_color(cloud_aligned, 0, 0, 255);
    viewer->addPointCloud<PointT>(cloud_aligned, aligned_color, "aligned cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned cloud");

    viewer->addText("Source (Red), Target (Green), Aligned (Blue)", 10, 10, "info_text", 0);

    viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0); // Default camera position
    viewer->spin(); // Keep viewer open until closed manually
}


int main(int argc, char** argv) {
    // Set spdlog to console output with info level
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    if (argc != 3) {
        spdlog::error("Usage: {} <source_pcd_file> <target_pcd_file>", argv[0]);
        spdlog::info("Example: ./ndt_registration scan_01.pcd scan_02.pcd");
        return 1;
    }

    std::string source_filename = argv[1];
    std::string target_filename = argv[2];

    pcl::PointCloud<PointT>::Ptr cloud_source(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_target(new pcl::PointCloud<PointT>());

    // --- 1. Load the input point clouds ---
    spdlog::info("Loading source point cloud from: {}", source_filename);
    if (pcl::io::loadPCDFile<PointT>(source_filename, *cloud_source) == -1) {
        spdlog::error("Couldn't read source file {}. Please ensure it's a valid PCD.", source_filename);
        return -1;
    }
    spdlog::info("Loaded source cloud with {} points.", cloud_source->size());

    spdlog::info("Loading target point cloud from: {}", target_filename);
    if (pcl::io::loadPCDFile<PointT>(target_filename, *cloud_target) == -1) {
        spdlog::error("Couldn't read target file {}. Please ensure it's a valid PCD.", target_filename);
        return -1;
    }
    spdlog::info("Loaded target cloud with {} points.", cloud_target->size());

    // --- 2. (Optional but Recommended) Downsample the point clouds ---
    // NDT can be computationally expensive for very dense clouds.
    // Downsampling helps speed up the process while maintaining sufficient detail.
    pcl::PointCloud<PointT>::Ptr filtered_source(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr filtered_target(new pcl::PointCloud<PointT>());

    float voxel_size = 0.5f; // Adjust this value based on your point cloud density and desired accuracy
    pcl::ApproximateVoxelGrid<PointT> approximate_voxel_filter;
    approximate_voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);

    approximate_voxel_filter.setInputCloud(cloud_source);
    approximate_voxel_filter.filter(*filtered_source);
    spdlog::info("Filtered source cloud from {} to {} points using voxel size {}.",
                 cloud_source->size(), filtered_source->size(), voxel_size);

    approximate_voxel_filter.setInputCloud(cloud_target);
    approximate_voxel_filter.filter(*filtered_target);
    spdlog::info("Filtered target cloud from {} to {} points using voxel size {}.",
                 cloud_target->size(), filtered_target->size(), voxel_size);

    // --- 3. Set up the NDT registration algorithm ---
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;

    // Set NDT parameters (these are crucial for performance and accuracy)
    // Resolution of the voxel grid (smaller = more detailed, slower)
    ndt.setResolution(1.0); // Common range: 0.5 to 2.0 meters

    // Set maximum number of registration iterations
    ndt.setMaximumIterations(100);

    // Set the transformation epsilon (maximum transformation difference between iterations for convergence)
    ndt.setTransformationEpsilon(0.01); // Converge if transformation difference is less than 1cm

    // Set the step size (for line search in optimization)
    ndt.setStepSize(0.1); // How much to step along the gradient direction

    // Set the target cloud (the cloud that the source will be aligned to)
    ndt.setInputTarget(filtered_target);

    // --- 4. Perform the registration ---
    pcl::PointCloud<PointT>::Ptr cloud_aligned(new pcl::PointCloud<PointT>());

    spdlog::info("Starting NDT registration...");
    auto start_time = std::chrono::high_resolution_clock::now();

    // Set the source cloud and provide an initial guess (identity matrix in this case)
    // A good initial guess (e.g., from IMU or odometry) significantly improves performance!
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    // Example of a small perturbation for testing:
    // initial_guess(0,3) = 0.5; // Translate 0.5m in X
    // initial_guess(1,3) = 0.2; // Translate 0.2m in Y
    
    ndt.setInputSource(filtered_source);
    ndt.align(*cloud_aligned, initial_guess); // Perform registration

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    spdlog::info("NDT registration finished in {} ms.", duration.count());

    // --- 5. Get results ---
    if (ndt.hasConverged()) {
        spdlog::info("NDT converged.");
        spdlog::info("Transformation score: {:.4f}", ndt.getFitnessScore());
        
        Eigen::Matrix4f final_transformation = ndt.getFinalTransformation();
        spdlog::info("Final Transformation Matrix, qx {} qy {}, qz {}, qw {}",
                     final_transformation(0, 3), final_transformation(1, 3),
                     final_transformation(2, 3), final_transformation(3, 3));
                     

        // Transform the original (unfiltered) source cloud with the found transformation
        pcl::transformPointCloud(*cloud_source, *cloud_aligned, final_transformation);

    } else {
        spdlog::warn("NDT did NOT converge after {} iterations.", ndt.getMaximumIterations());
        spdlog::warn("Transformation score (last iteration): {:.4f}", ndt.getFitnessScore());
        // In this case, cloud_aligned will contain the result of the last iteration.
    }

    // --- 6. Save the aligned cloud ---
    std::string output_filename = "aligned_cloud_output.pcd";
    pcl::io::savePCDFileBinary(output_filename, *cloud_aligned);
    spdlog::info("Saved aligned cloud to: {} ({} points).", output_filename, cloud_aligned->size());

    // --- 7. (Optional) Visualize the results ---
    // Requires PCL visualization module. If not installed, you can comment this out.
    visualizeClouds(cloud_source, cloud_target, cloud_aligned);

    return 0;
}