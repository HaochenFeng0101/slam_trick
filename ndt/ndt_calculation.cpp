#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <limits> // For std::numeric_limits
#include <iomanip>
#include <sstream>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h> 

// --- Point Type (similar to PCL's PointXYZ) ---
struct MyPoint {
    float x, y, z;

    Eigen::Vector3f toEigenVector() const {
        return Eigen::Vector3f(x, y, z);
    }
};

// --- Voxel Structure for NDT Target Grid ---
struct NDT_Voxel {
    Eigen::Vector3f mean;
    Eigen::Matrix3f covariance;
    Eigen::Matrix3f inv_covariance;
    int num_points;

    NDT_Voxel() : mean(Eigen::Vector3f::Zero()), covariance(Eigen::Matrix3f::Zero()),
                  inv_covariance(Eigen::Matrix3f::Identity()), num_points(0) {}
};

// --- Custom comparator for Eigen::Vector3i to use with std::map ---
struct VoxelIndexComparator {
    bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const {
        if (a.x() != b.x()) {
            return a.x() < b.x();
        }
        if (a.y() != b.y()) {
            return a.y() < b.y();
        }
        return a.z() < b.z();
    }
};

// --- Helper Functions for Transformation ---

Eigen::Matrix3f eulerToRotationMatrix(float roll, float pitch, float yaw) {
    Eigen::Matrix3f R_x;
    R_x << 1, 0, 0,
           0, std::cos(roll), -std::sin(roll),
           0, std::sin(roll), std::cos(roll);

    Eigen::Matrix3f R_y;
    R_y << std::cos(pitch), 0, std::sin(pitch),
           0, 1, 0,
           -std::sin(pitch), 0, std::cos(pitch);

    Eigen::Matrix3f R_z;
    R_z << std::cos(yaw), -std::sin(yaw), 0,
           std::sin(yaw), std::cos(yaw), 0,
           0, 0, 1;

    return R_z * R_y * R_x; 
}

// This function creates a 4x4 homogeneous transformation matrix
Eigen::Affine3f createHomogeneousTransform(const Eigen::VectorXf& transform_params) {
    float tx = transform_params(0);
    float ty = transform_params(1);
    float tz = transform_params(2);
    float roll = transform_params(3);
    float pitch = transform_params(4);
    float yaw = transform_params(5);

    Eigen::Matrix3f R = eulerToRotationMatrix(roll, pitch, yaw);
    Eigen::Vector3f t(tx, ty, tz);

    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.linear() = R;
    T.translation() = t;
    return T;
}

Eigen::Vector3f transformPoint(const MyPoint& p, const Eigen::VectorXf& transform_params) {
    float tx = transform_params(0);
    float ty = transform_params(1);
    float tz = transform_params(2);
    float roll = transform_params(3);
    float pitch = transform_params(4);
    float yaw = transform_params(5);

    Eigen::Matrix3f R = eulerToRotationMatrix(roll, pitch, yaw);
    Eigen::Vector3f t(tx, ty, tz);
    Eigen::Vector3f p_eigen = p.toEigenVector();

    return R * p_eigen + t;
}

std::map<Eigen::Vector3i, NDT_Voxel, VoxelIndexComparator> buildTargetNDTGrid(const std::vector<MyPoint>& cloud, float resolution) {
    std::map<Eigen::Vector3i, std::vector<MyPoint>, VoxelIndexComparator> voxel_points;
    std::map<Eigen::Vector3i, NDT_Voxel, VoxelIndexComparator> ndt_grid;

    for (const auto& p : cloud) {
        Eigen::Vector3i voxel_idx(
            static_cast<int>(std::floor(p.x / resolution)),
            static_cast<int>(std::floor(p.y / resolution)),
            static_cast<int>(std::floor(p.z / resolution))
        );
        voxel_points[voxel_idx].push_back(p);
    }

    float regularization_factor = 1e-6f;
    int skipped_voxels_too_few_points = 0; // For debugging
    for (const auto& pair : voxel_points) {
        const Eigen::Vector3i& voxel_idx = pair.first;
        const std::vector<MyPoint>& points_in_voxel = pair.second;

        if (points_in_voxel.size() < 3) {
            skipped_voxels_too_few_points++; // Increment counter
            continue;
        }

        NDT_Voxel voxel_data;
        voxel_data.num_points = points_in_voxel.size();

        Eigen::Vector3f sum_points = Eigen::Vector3f::Zero();
        for (const auto& p : points_in_voxel) {
            sum_points += p.toEigenVector();
        }
        voxel_data.mean = sum_points / static_cast<float>(voxel_data.num_points);

        Eigen::Matrix3f cov_matrix = Eigen::Matrix3f::Zero();
        for (const auto& p : points_in_voxel) {
            Eigen::Vector3f centered_p = p.toEigenVector() - voxel_data.mean;
            cov_matrix += centered_p * centered_p.transpose();
        }
        // Use N-1 for unbiased sample covariance if N > 1, else use 1 to avoid division by zero
        cov_matrix /= static_cast<float>(voxel_data.num_points > 1 ? voxel_data.num_points - 1 : 1);

        // Add regularization to prevent singular covariance matrices
        voxel_data.covariance = cov_matrix + regularization_factor * Eigen::Matrix3f::Identity();
        
        // Ensure covariance is invertible (can be an issue if points are collinear/coplanar in a voxel)
        voxel_data.inv_covariance = voxel_data.covariance.inverse();
        
        ndt_grid[voxel_idx] = voxel_data;
    }
    spdlog::info("Built NDT grid with {} populated voxels.", ndt_grid.size());
    if (skipped_voxels_too_few_points > 0) {
        spdlog::warn("Skipped {} voxels during NDT grid building due to having fewer than 3 points.", skipped_voxels_too_few_points);
    }
    return ndt_grid;
}

void calculateSinglePointNDTDifferentials(
    const Eigen::Vector3f& p_transformed,
    const MyPoint& p_original,
    const Eigen::Vector3f& voxel_mean,
    const Eigen::Matrix3f& voxel_inv_covariance,
    const Eigen::VectorXf& transform_params,
    double& objective_contribution,
    Eigen::VectorXf& gradient_contribution,
    Eigen::MatrixXf& hessian_contribution)
{
    float roll = transform_params(3);
    float pitch = transform_params(4);
    float yaw = transform_params(5);

    Eigen::Vector3f q = p_transformed - voxel_mean;

    objective_contribution = 0.5 * q.transpose() * voxel_inv_covariance * q;

    Eigen::MatrixXf J(3, 6);
    Eigen::Vector3f p_orig_eigen = p_original.toEigenVector();

    // d(Rx)/dx, d(Ry)/dy, d(Rz)/dz partial derivatives
    J.col(0) << 1, 0, 0; // d(tx)/dx = 1
    J.col(1) << 0, 1, 0; // d(ty)/dy = 1
    J.col(2) << 0, 0, 1; // d(tz)/dz = 1

    // Partial derivatives with respect to roll, pitch, yaw
    // These are simplified for clarity but should be exact for the chosen Euler angle convention (Z-Y-X)
    // The derivatives of the rotation matrix R = R_z * R_y * R_x
    
    Eigen::Matrix3f R_x_mat = eulerToRotationMatrix(roll, 0, 0);
    Eigen::Matrix3f R_y_mat = eulerToRotationMatrix(0, pitch, 0);
    Eigen::Matrix3f R_z_mat = eulerToRotationMatrix(0, 0, yaw);

    // dR/d(roll) = R_z * R_y * dR_x/d(roll)
    Eigen::Matrix3f dR_droll; 
    dR_droll << 0, 0, 0,
                0, -std::sin(roll), -std::cos(roll),
                0, std::cos(roll), -std::sin(roll);
    dR_droll = R_z_mat * R_y_mat * dR_droll; // Correct chain rule application
    J.col(3) = dR_droll * p_orig_eigen; 

    // dR/d(pitch) = R_z * dR_y/d(pitch) * R_x
    Eigen::Matrix3f dR_dpitch; 
    dR_dpitch << -std::sin(pitch), 0, std::cos(pitch),
                 0, 0, 0,
                 -std::cos(pitch), 0, -std::sin(pitch);
    dR_dpitch = R_z_mat * dR_dpitch * R_x_mat; // Correct chain rule application
    J.col(4) = dR_dpitch * p_orig_eigen; 

    // dR/d(yaw) = dR_z/d(yaw) * R_y * R_x
    Eigen::Matrix3f dR_dyaw; 
    dR_dyaw << -std::sin(yaw), -std::cos(yaw), 0,
               std::cos(yaw), -std::sin(yaw), 0,
               0, 0, 0;
    dR_dyaw = dR_dyaw * R_y_mat * R_x_mat; // Correct chain rule application
    J.col(5) = dR_dyaw * p_orig_eigen; 

    gradient_contribution = J.transpose() * voxel_inv_covariance * q;
    hessian_contribution = J.transpose() * voxel_inv_covariance * J;
}

struct NDTIterationResult {
    double total_objective_score;
    Eigen::VectorXf gradient;
    Eigen::MatrixXf hessian;
    bool success;
};

NDTIterationResult performSingleNDTIteration(
    const std::vector<MyPoint>& source_cloud,
    const std::map<Eigen::Vector3i, NDT_Voxel, VoxelIndexComparator>& target_ndt_grid,
    const Eigen::VectorXf& current_transform_params,
    float resolution)
{
    NDTIterationResult result;
    result.total_objective_score = 0.0;
    result.gradient = Eigen::VectorXf::Zero(6);
    result.hessian = Eigen::MatrixXf::Zero(6, 6);
    result.success = false;

    if (source_cloud.empty()) {
        spdlog::warn("Source cloud is empty for NDT iteration.");
        return result;
    }
    if (target_ndt_grid.empty()) {
        spdlog::warn("Target NDT grid is empty. Cannot perform iteration.");
        return result;
    }

    int num_points_processed = 0;
    for (const auto& p_original : source_cloud) {
        Eigen::Vector3f p_transformed = transformPoint(p_original, current_transform_params);

        Eigen::Vector3i voxel_idx(
            static_cast<int>(std::floor(p_transformed.x() / resolution)),
            static_cast<int>(std::floor(p_transformed.y() / resolution)),
            static_cast<int>(std::floor(p_transformed.z() / resolution))
        );

        auto it = target_ndt_grid.find(voxel_idx);
        if (it != target_ndt_grid.end()) {
            const NDT_Voxel& voxel_data = it->second;

            double obj_contrib;
            Eigen::VectorXf grad_contrib(6);
            Eigen::MatrixXf hess_contrib(6, 6);

            calculateSinglePointNDTDifferentials(
                p_transformed, p_original,
                voxel_data.mean, voxel_data.inv_covariance,
                current_transform_params,
                obj_contrib, grad_contrib, hess_contrib
            );

            result.total_objective_score += obj_contrib;
            result.gradient += grad_contrib;
            result.hessian += hess_contrib;
            num_points_processed++;
        }
    }

    if (num_points_processed == 0) {
        spdlog::warn("No source points found in any target NDT voxel. Check initial guess or overlap.");
        result.success = false;
    } else {
        spdlog::info("Processed {} / {} source points in this iteration.", num_points_processed, source_cloud.size());
        result.success = true;
    }
    return result;
}

// --- Helper to convert PCL PointCloud<PointXYZ> to std::vector<MyPoint> ---
std::vector<MyPoint> convertPCLPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud) {
    std::vector<MyPoint> my_points;
    my_points.reserve(pcl_cloud->points.size());
    for (const auto& p : pcl_cloud->points) {
        my_points.push_back({p.x, p.y, p.z});
    }
    return my_points;
}

// --- Main Function (for testing and demonstration) ---
int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    if (argc < 3) {
        spdlog::error("Usage: {} <target_pcd_file> <source_pcd_file>", argv[0]);
        return 1;
    }

    std::string target_pcd_path = argv[1];
    std::string source_pcd_path = argv[2];

    // --- 1. Load point clouds from files using PCL's native PointXYZ type ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_transformed_source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // Load the PCD files into PCL's standard PointXYZ format
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_pcd_path, *pcl_target_cloud) == -1) {
        spdlog::error("Could not read target PCD file: {}. Exiting.", target_pcd_path);
        return 1;
    }
    spdlog::info("Loaded {} points from {}.", pcl_target_cloud->points.size(), target_pcd_path);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(source_pcd_path, *pcl_source_cloud) == -1) {
        spdlog::error("Could not read source PCD file: {}. Exiting.", source_pcd_path);
        return 1;
    }
    spdlog::info("Loaded {} points from {}.", pcl_source_cloud->points.size(), source_pcd_path);

    // Make a copy of the source cloud to transform iteratively
    *pcl_transformed_source_cloud = *pcl_source_cloud;

    // Convert the PCL target cloud to your custom MyPoint vectors (only once as it's static)
    std::vector<MyPoint> target_cloud = convertPCLPointCloud(pcl_target_cloud);

    // --- 2. Set NDT resolution (ADJUST THIS BASED ON YOUR POINT CLOUD SCALE) ---
    float ndt_resolution = 0.1f; 

    // --- 3. Build NDT grid for the target cloud ---
    spdlog::info("Building NDT grid for target cloud with resolution: {}m", ndt_resolution);
    std::map<Eigen::Vector3i, NDT_Voxel, VoxelIndexComparator> target_ndt_grid = buildTargetNDTGrid(target_cloud, ndt_resolution);

    // --- 4. Set initial guess for transformation parameters ---
    Eigen::VectorXf current_transform_params(6);
    current_transform_params << 0.0f, 0.0f, 0.0f, // tx, ty, tz (meters)
                                0.0f, 0.0f, 0.0f; // roll, pitch, yaw (radians)
    
    // --- 5. Perform NDT iteratively with Levenberg-Marquardt ---
    const int max_iterations = 50; 
    const double convergence_threshold = 1e-5; 

    // Levenberg-Marquardt parameters
    double lambda = 0.01; // Initial damping parameter (adjust as needed, start small for Newton-like)
    const double lambda_factor_increase = 10.0; // Factor to increase lambda if step fails
    const double lambda_factor_decrease = 0.1;  // Factor to decrease lambda if step succeeds
    const double max_lambda = 1e10; // Upper bound for lambda
    const double min_lambda = 1e-10; // Lower bound for lambda

    double current_objective_score = std::numeric_limits<double>::max(); // Initial high score

    for (int iter = 0; iter < max_iterations; ++iter) {
        spdlog::info("--- NDT Iteration {} ---", iter + 1);
        spdlog::info("Current Transform Guess: [tx:{:.6f}, ty:{:.6f}, tz:{:.6f}, roll:{:.6f}, pitch:{:.6f}, yaw:{:.6f}]",
                     current_transform_params(0), current_transform_params(1), current_transform_params(2),
                     current_transform_params(3), current_transform_params(4), current_transform_params(5));

        // Convert the current state of pcl_transformed_source_cloud for NDT calculation
        std::vector<MyPoint> current_source_cloud_for_ndt = convertPCLPointCloud(pcl_transformed_source_cloud);

        NDTIterationResult result = performSingleNDTIteration(
            current_source_cloud_for_ndt, target_ndt_grid, current_transform_params, ndt_resolution
        );

        if (!result.success) {
            spdlog::error("NDT iteration failed (no points in voxels). Breaking loop.");
            break;
        }
        
        // This is the objective score based on current_transform_params *before* the update
        current_objective_score = result.total_objective_score;
        spdlog::info("  Current Objective Score: {:.6f}", current_objective_score);
        
        Eigen::VectorXf gradient = result.gradient;
        Eigen::MatrixXf hessian = result.hessian;

        // Check for ill-conditioned Hessian BEFORE adding lambda
        bool is_hessian_bad = false;
        if (hessian.hasNaN() || !hessian.allFinite()) { 
            is_hessian_bad = true;
            spdlog::warn("  Raw Hessian contains NaN/Inf values.");
        } else if (hessian.norm() < std::numeric_limits<float>::epsilon() * 1e3) { 
            is_hessian_bad = true;
            spdlog::warn("  Raw Hessian norm is near zero, indicating ill-conditioning.");
        }

        Eigen::VectorXf delta_x;
        if (is_hessian_bad) {
            // If the raw Hessian is completely bad, LM might not even help, or it might just take tiny steps.
            // In a more robust implementation, you might still try LM with a large lambda,
            // or switch to pure gradient descent for a few steps.
            spdlog::error("Hessian is ill-conditioned. Cannot compute reliable Levenberg-Marquardt step. Breaking loop.");
            break;
        }

        // --- Levenberg-Marquardt Step Calculation ---
        Eigen::MatrixXf JtJ_damped = hessian;
        for (int i = 0; i < 6; ++i) {
            JtJ_damped(i, i) += lambda * JtJ_damped(i,i); // Damping uses diagonal elements
            // Alternatively: JtJ_damped(i, i) += lambda; // Damping with simple scalar
        }

        // Check if damped matrix is invertible. This is crucial for robustness.
        Eigen::FullPivLU<Eigen::MatrixXf> lu(JtJ_damped);
        if (!lu.isInvertible()) {
            spdlog::error("Damped Hessian is not invertible. Increasing lambda and retrying in next iteration. Breaking loop.");
            lambda = std::min(lambda * lambda_factor_increase, max_lambda);
            continue; // Skip the rest of this iteration, try again with higher lambda
        }

        delta_x = -lu.solve(gradient); // Solve JtJ_damped * delta_x = -gradient
        
        spdlog::info("  Calculated Delta X (update step norm): {:.6f}", delta_x.norm());

        // --- Evaluate New State ---
        Eigen::VectorXf next_transform_params = current_transform_params + delta_x;

        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_candidate_transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        Eigen::Affine3f candidate_transform_matrix = createHomogeneousTransform(next_transform_params);
        pcl::transformPointCloud(*pcl_source_cloud, *temp_candidate_transformed_cloud, candidate_transform_matrix);
        
        std::vector<MyPoint> candidate_source_cloud_for_ndt = convertPCLPointCloud(temp_candidate_transformed_cloud);
        
        NDTIterationResult candidate_result = performSingleNDTIteration(
            candidate_source_cloud_for_ndt, target_ndt_grid, next_transform_params, ndt_resolution
        );

        if (!candidate_result.success) {
            spdlog::warn("Candidate transform resulted in no points in voxels. Increasing lambda.");
            lambda = std::min(lambda * lambda_factor_increase, max_lambda);
            continue; // Try again with a higher lambda in the next iteration
        }

        double new_objective_score = candidate_result.total_objective_score;

        // --- Adaptive Damping Parameter Update ---
        if (new_objective_score < current_objective_score) {
            // Step improved the objective: decrease lambda (move towards Newton)
            current_transform_params = next_transform_params;
            *pcl_transformed_source_cloud = *temp_candidate_transformed_cloud; // Update source cloud for next iter
            lambda = std::max(lambda * lambda_factor_decrease, min_lambda);
            spdlog::info("  Objective improved: {:.6f} -> {:.6f}. Decreasing lambda to {:.6f}", 
                         current_objective_score, new_objective_score, lambda);
        } else {
            // Step did NOT improve the objective: increase lambda (move towards Gradient Descent)
            lambda = std::min(lambda * lambda_factor_increase, max_lambda);
            spdlog::warn("  Objective did not improve: {:.6f} -> {:.6f}. Increasing lambda to {:.6f}", 
                         current_objective_score, new_objective_score, lambda);
            // Do NOT update current_transform_params or pcl_transformed_source_cloud
        }

        // Check for convergence based on delta_x norm
        if (delta_x.norm() < convergence_threshold) {
            spdlog::info("Converged! Delta X norm below threshold.");
            break;
        }

       
    }
    
    spdlog::info("Final estimated transform: [tx:{:.6f}, ty:{:.6f}, tz:{:.6f}, roll:{:.6f}, pitch:{:.6f}, yaw:{:.6f}]",
                 current_transform_params(0), current_transform_params(1), current_transform_params(2),
                 current_transform_params(3), current_transform_params(4), current_transform_params(5));

    // Save the final transformed source cloud
    pcl::io::savePCDFileASCII("ndt_final_transformed_source.pcd", *pcl_transformed_source_cloud);
    spdlog::info("Saved final transformed source cloud to ndt_final_transformed_source.pcd");
    
    return 0;
}