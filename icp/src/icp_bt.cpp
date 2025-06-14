#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm> // For std::min_element, std::max_element
#include <numeric>   // For std::accumulate
#include <iomanip>   // For std::setprecision

// Eigen for linear algebra
#include <Eigen/Dense> // For Matrix, Vector, SVD

// PCL Headers for Point Clouds, KdTree, and I/O
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h> // For the optimized KdTree
#include <pcl/io/pcd_io.h>           // For loading/saving PCD files
#include <pcl/io/ply_io.h>           // For loading PLY files
#include <pcl/common/transforms.h>   // For pcl::transformPointCloud

// spdlog for logging
#include <spdlog/spdlog.h>



// --- SVD-based Transformation Estimation (Kabsch Algorithm) ---
Eigen::Matrix4f estimateTransformationSVD(
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& correspondences
) {
    if (correspondences.empty()) {
        spdlog::warn("No correspondences provided for SVD estimation. Returning Identity.");
        return Eigen::Matrix4f::Identity();
    }

    // 1. Calculate centroids
    Eigen::Vector3f p_centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f q_centroid = Eigen::Vector3f::Zero();

    for (const auto& pair : correspondences) {
        p_centroid += pair.first;
        q_centroid += pair.second;
    }

    p_centroid /= static_cast<float>(correspondences.size());
    q_centroid /= static_cast<float>(correspondences.size());

    // 2. Compute centered point sets
    std::vector<Eigen::Vector3f> p_centered(correspondences.size());
    std::vector<Eigen::Vector3f> q_centered(correspondences.size());

    for (size_t i = 0; i < correspondences.size(); ++i) {
        p_centered[i] = correspondences[i].first - p_centroid;
        q_centered[i] = correspondences[i].second - q_centroid;
    }

    // 3. Compute the covariance matrix H
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < correspondences.size(); ++i) {
        H += p_centered[i] * q_centered[i].transpose();
    }
    
    // 4. Perform SVD on H: H = U * S * V^T
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    // 5. Compute Rotation Matrix R
    Eigen::Matrix3f R = V * U.transpose();

    // Special reflection case handling
    if (R.determinant() < 0) {
        spdlog::warn("SVD resulted in a reflection. Correcting.");
        V.col(2) *= -1; // Correct the reflection
        R = V * U.transpose(); // Recompute R
    }

    // 6. Compute Translation Vector t
    Eigen::Vector3f t = q_centroid - R * p_centroid;

    // 7. Assemble the 4x4 homogeneous transformation matrix
    Eigen::Matrix4f T_matrix = Eigen::Matrix4f::Identity();
    T_matrix.block<3, 3>(0, 0) = R;
    T_matrix.block<3, 1>(0, 3) = t;

    return T_matrix;
}

// --- ICP Algorithm using PCL's KdTree ---
Eigen::Matrix4f icp(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_initial, // The cloud we want to align
    const pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,       // The reference cloud
    float max_correspondence_distance,       // Max distance for a valid correspondence
    int max_iterations,                      // Max ICP iterations
    float transformation_epsilon             // Minimum change in transformation to stop
) {
    if (source_cloud_initial->empty() || target_cloud->empty()) {
        spdlog::error("Input clouds cannot be empty. Returning Identity transform.");
        return Eigen::Matrix4f::Identity();
    }

    Eigen::Matrix4f current_total_transform = Eigen::Matrix4f::Identity(); // Accumulated transformation
    // Create a copy of the source cloud that will be iteratively transformed
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    *transformed_source_cloud = *source_cloud_initial; // Start with the initial source cloud

    // Build Kd-Tree for the target cloud (built once)
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target_cloud);

    spdlog::info("Starting ICP...");

    for (int iter = 0; iter < max_iterations; ++iter) {
        spdlog::info("--- Iteration {}/{} ---", iter + 1, max_iterations);
        // std::cout << std::fixed << std::setprecision(6); // Set precision for output

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> correspondences;
        float current_mean_squared_error = 0.0f;
        int num_valid_correspondences = 0;

        // Correspondence Search (point to point)
        for (const auto& p_i : transformed_source_cloud->points) { // p_i is in the 'current_source_frame'
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);

            // Perform nearest neighbor search for p_i in target_cloud
            if (kdtree.nearestKSearch(p_i, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                // Check if the closest point found is within the maximum correspondence distance
                if (pointNKNSquaredDistance[0] <= max_correspondence_distance * max_correspondence_distance) {
                    pcl::PointXYZ qb_k_pcl = target_cloud->points[pointIdxNKNSearch[0]];
                    Eigen::Vector3f qb_k_eigen = qb_k_pcl.getVector3fMap();
                    Eigen::Vector3f p_i_eigen_transformed = p_i.getVector3fMap(); // This is the point in the *current* transformed cloud

                    // The SVD algorithm needs pairs (P_current, Q_target)
                    // P_current is p_i_eigen_transformed, and Q_target is qb_k_eigen
                    correspondences.push_back({p_i_eigen_transformed, qb_k_eigen});
                    current_mean_squared_error += pointNKNSquaredDistance[0];
                    num_valid_correspondences++;
                }
            }
        }

        if (correspondences.empty()) {
            spdlog::warn("No correspondences found in iteration {}. Stopping ICP. This often means the initial alignment is too poor or max_correspondence_distance is too small.", iter + 1);
            break;
        }

        current_mean_squared_error /= static_cast<float>(num_valid_correspondences);
        spdlog::info("Mean Squared Error: {:.6f}, Valid correspondences: {}",
                    current_mean_squared_error, num_valid_correspondences);
        
        // 2. Transformation Estimation using SVD
        // This 'transformation_delta' maps 'transformed_source_cloud' to 'target_cloud'
        Eigen::Matrix4f transformation_delta = estimateTransformationSVD(correspondences);

        // 3. Apply transformation to the transformed_source_cloud
        // This moves transformed_source_cloud closer to target_cloud
        pcl::transformPointCloud(*transformed_source_cloud, *transformed_source_cloud, transformation_delta);

        // Accumulate the transformation
        // The new total transform is T_delta * T_old_total
        // This means: (new_src_point) = T_delta * (old_src_point)
        // And (old_src_point) = T_old_total * (initial_src_point)
        // So (new_src_point) = T_delta * T_old_total * (initial_src_point)
        current_total_transform = transformation_delta * current_total_transform;

        // 4. Check for convergence
        // The Frobenius norm of the difference between the current delta transform and identity
        float transform_change = (transformation_delta - Eigen::Matrix4f::Identity()).norm();
        spdlog::info("Transformation Change (Frobenius norm of (Delta_T - I)): {:.6f}", transform_change);
        
        if (transform_change < transformation_epsilon) {
            spdlog::info("ICP converged after {} iterations.", iter + 1);
            break;
        }
    }

    spdlog::info("ICP finished.");
    return current_total_transform;
}

// Helper function to print a transformation matrix
void printMatrix(const Eigen::Matrix4f& matrix, const std::string& name) {
    std::cout << name << ":\n";
    std::cout << std::fixed << std::setprecision(6) << matrix << "\n\n";
}

// --- Main function to load PCDs/PLYs and run ICP ---
int main(int argc, char** argv) {
    // Set spdlog to console output with info level
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); // More detailed pattern

    if (argc != 3) {
        spdlog::error("Usage: {} <source_file> <target_file>", argv[0]);
        return 1;
    }

    // Load source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::string source_filename = argv[1];
    std::string target_filename = argv[2];

    // Determine file type and load
    auto load_cloud = [](const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) -> bool {
        if (filename.rfind(".pcd") == filename.length() - 4) {
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
                spdlog::error("Couldn't read PCD file: {}", filename);
                return false;
            }
        } else if (filename.rfind(".ply") == filename.length() - 4) {
            if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, *cloud) == -1) {
                spdlog::error("Couldn't read PLY file: {}", filename);
                return false;
            }
        } else {
            spdlog::error("Unsupported file format for {}. Please use .pcd or .ply", filename);
            return false;
        }
        return true;
    };

    if (!load_cloud(source_filename, source_cloud)) return -1;
    if (!load_cloud(target_filename, target_cloud)) return -1;
    
    spdlog::info("Loaded source cloud from {} with {} points.", source_filename, source_cloud->size());
    spdlog::info("Loaded target cloud from {} with {} points.", target_filename, target_cloud->size());

    // Debug: Print first and last points after loading
    if (!source_cloud->empty()) {
        spdlog::info("Source cloud first point: ({:.6f}, {:.6f}, {:.6f})", 
                     source_cloud->points[0].x, source_cloud->points[0].y, source_cloud->points[0].z);
        if (source_cloud->size() > 1) {
            spdlog::info("Source cloud last point: ({:.6f}, {:.6f}, {:.6f})", 
                         source_cloud->points[source_cloud->size()-1].x, source_cloud->points[source_cloud->size()-1].y, source_cloud->points[source_cloud->size()-1].z);
        }
    } else {
        spdlog::warn("Source cloud is empty after loading!");
    }
    if (!target_cloud->empty()) {
        spdlog::info("Target cloud first point: ({:.6f}, {:.6f}, {:.6f})", 
                     target_cloud->points[0].x, target_cloud->points[0].y, target_cloud->points[0].z);
        if (target_cloud->size() > 1) {
            spdlog::info("Target cloud last point: ({:.6f}, {:.6f}, {:.6f})", 
                         target_cloud->points[target_cloud->size()-1].x, target_cloud->points[target_cloud->size()-1].y, target_cloud->points[target_cloud->size()-1].z);
        }
    } else {
        spdlog::warn("Target cloud is empty after loading!");
    }


    // ICP Parameters
    // Adjust these based on the scale of your data and expected misalignment
    float max_correspondence_distance = 0.05f; // Max distance for a valid pair in meters (adjusted for typical bunny scale)
    int max_iterations = 100;                  // Max iterations for ICP
    float transformation_epsilon = 1e-6f;     // Minimum change in transformation to stop

    spdlog::info("\n--- Running ICP ---");
    spdlog::info("Max Correspondence Distance: {:.3f}m", max_correspondence_distance);
    spdlog::info("Max Iterations: {}", max_iterations);
    spdlog::info("Transformation Epsilon: {:.6f}", transformation_epsilon);

    Eigen::Matrix4f final_transform_source_to_target = icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance,
        max_iterations,
        transformation_epsilon
    );

    printMatrix(final_transform_source_to_target, "Final Estimated Transformation (Source -> Target)");

    // Save the aligned source cloud to a new PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*source_cloud, *aligned_source_cloud, final_transform_source_to_target);

    std::string output_filename = "/home/haochen/code/slam_trick/icp/results/bunny/result.pcd"; // Changed output filename
    pcl::io::savePCDFileASCII(output_filename, *aligned_source_cloud);
    spdlog::info("Aligned source cloud saved to: {}", output_filename);

    return 0;
}