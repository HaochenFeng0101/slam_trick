#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spdlog/spdlog.h> // For logging

#include "feature_extractor.hpp" 

// Using PointXYZ for simplicity, matches FeatureExtractor
using PointT = pcl::PointXYZ;

int main(int argc, char** argv) {
    // Set spdlog to console output with info level and pattern
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    if (argc < 2) {
        spdlog::error("Usage: {} <input_pcd_file>", argv[0]);
        spdlog::info("Example: ./lidar_feature_extractor /path/to/your/scan.pcd");
        return 1;
    }

    std::string input_filename = argv[1];
    pcl::PointCloud<PointT>::Ptr input_cloud(new pcl::PointCloud<PointT>());

    // Load the input PCD file
    if (pcl::io::loadPCDFile<PointT>(input_filename, *input_cloud) == -1) {
        spdlog::error("Couldn't read file {}. Please ensure it's a valid PCD.", input_filename);
        return -1;
    }

    spdlog::info("Loaded input cloud from {}: {} points.", input_filename, input_cloud->size());

    // Create an instance of FeatureExtractor
    FeatureExtractor extractor;

    // Output clouds for features
    pcl::PointCloud<PointT>::Ptr edge_features(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr surface_features(new pcl::PointCloud<PointT>());

    // Extract features
    extractor.extractFeatures(input_cloud, edge_features, surface_features);

    // Prepare output filenames
    size_t last_dot_pos = input_filename.find_last_of(".");
    std::string base_filename = (last_dot_pos == std::string::npos) ? input_filename : input_filename.substr(0, last_dot_pos);
    
    std::string edge_output_filename = base_filename + "_edges.pcd";
    std::string surface_output_filename = base_filename + "_surfaces.pcd";

    // Save the extracted features as binary PCD files
    pcl::io::savePCDFileBinary(edge_output_filename, *edge_features);
    spdlog::info("Saved edge features to {}: {} points.", edge_output_filename, edge_features->size());

    pcl::io::savePCDFileBinary(surface_output_filename, *surface_features);
    spdlog::info("Saved surface features to {}: {} points.", surface_output_filename, surface_features->size());

    spdlog::info("Feature extraction complete.");

    return 0;
}