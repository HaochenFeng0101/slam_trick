#include "feature_extractor.hpp"
#include <spdlog/spdlog.h> // For logging
#include <pcl/common/centroid.h> // For computing centroid of a point cloud
#include <Eigen/Eigenvalues> // For SelfAdjointEigenSolver

FeatureExtractor::FeatureExtractor()
    : num_neighbors_(5), // Number of neighbors for local analysis (N+1 points, including the point itself)
      edge_threshold_(0.02f), // Tune these values. Lower value means more "flat" points classified as edges.
      surface_threshold_(0.0005f), // Higher value means more "rough" points classified as surfaces.
      NMS_radius_(0.15f) // 15 cm radius for Non-Maximum Suppression (tune as needed)
{
    spdlog::info("FeatureExtractor initialized with: Neighbors={}, EdgeThresh={:.4f}, SurfThresh={:.4f}, NMSRadius={:.4f}",
                 num_neighbors_, edge_threshold_, surface_threshold_, NMS_radius_);
}

std::vector<PointSmoothness> FeatureExtractor::computeSmoothness(
    const pcl::PointCloud<PointT>::Ptr& cloud)
{
    std::vector<PointSmoothness> point_smoothness;
    if (cloud->empty()) {
        spdlog::warn("Input cloud for smoothness computation is empty.");
        return point_smoothness;
    }

    // Need at least num_neighbors_ + 1 points to compute a meaningful covariance matrix
    if (cloud->size() < num_neighbors_ + 1) {
        spdlog::warn("Cloud size ({}) too small for {} neighbors. Cannot compute smoothness for all points.",
                     cloud->size(), num_neighbors_);
        // Assign max smoothness to all points if too few to process, essentially treating all as edges
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            point_smoothness.push_back({(int)i, std::numeric_limits<float>::max(), cloud->points[i]});
        }
        return point_smoothness;
    }

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    point_smoothness.reserve(cloud->size());

    for (size_t i = 0; i < cloud->points.size(); ++i) {
        std::vector<int> pointIdxKNNSearch;
        std::vector<float> pointKNNSquaredDistance;

        // Search for k = num_neighbors_ + 1 because the point itself might be returned.
        // We want 'num_neighbors_' *other* points.
        if (kdtree.nearestKSearch(cloud->points[i], num_neighbors_ + 1, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
            // Check if we actually found enough neighbors
            if (pointIdxKNNSearch.size() < num_neighbors_ + 1) {
                // Not enough neighbors found, this point is likely isolated or at cloud boundary.
                // Assign max smoothness (very sharp) so it's considered an edge if thresholds allow.
                point_smoothness.push_back({(int)i, std::numeric_limits<float>::max(), cloud->points[i]});
                continue;
            }

            // Create a temporary cloud of the neighborhood (including the central point)
            pcl::PointCloud<PointT>::Ptr neighborhood_cloud(new pcl::PointCloud<PointT>());
            neighborhood_cloud->points.reserve(pointIdxKNNSearch.size());
            for(int idx : pointIdxKNNSearch) {
                neighborhood_cloud->points.push_back(cloud->points[idx]);
            }

            // Compute centroid of the local neighborhood
            Eigen::Vector4f neighborhood_centroid_vec; // PCL's centroid function uses Eigen::Vector4f
            pcl::compute3DCentroid(*neighborhood_cloud, neighborhood_centroid_vec);
            Eigen::Vector3f neighborhood_centroid = neighborhood_centroid_vec.head<3>(); // Convert to Eigen::Vector3f

            // Compute covariance matrix of the local neighborhood
            Eigen::Matrix3f cov_matrix = Eigen::Matrix3f::Zero();
            for(const auto& p : neighborhood_cloud->points) {
                Eigen::Vector3f centered_p = p.getVector3fMap() - neighborhood_centroid;
                cov_matrix += centered_p * centered_p.transpose();
            }

            // Eigenvalue decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov_matrix);
            Eigen::Vector3f eigenvalues = es.eigenvalues();

            // Sort eigenvalues in ascending order (smallest to largest)
            // Eigen's eigenvalues are not guaranteed to be sorted for SelfAdjointEigenSolver
            std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());

            float lambda1 = eigenvalues(2); // Largest eigenvalue
            float lambda2 = eigenvalues(1); // Middle eigenvalue
            float lambda3 = eigenvalues(0); // Smallest eigenvalue

            float smoothness_val;

            // Curvature/smoothness measure:
            // The smallest eigenvalue (lambda3) indicates the variance along the direction perpendicular to the local surface.
            // A small lambda3 relative to the sum of eigenvalues suggests a flat (planar) surface.
            // A larger ratio (lambda3 / sum) suggests a less planar or more spherical/noisy neighborhood.
            // A common measure: (lambda3) / (lambda1 + lambda2 + lambda3)
            // Small value -> flat/smooth. Large value -> sharp/rough.
            if (lambda1 + lambda2 + lambda3 > 1e-6) { // Avoid division by zero for degenerate cases
                smoothness_val = lambda3 / (lambda1 + lambda2 + lambda3);
            } else {
                smoothness_val = std::numeric_limits<float>::max(); // Degenerate case, treat as very sharp
            }
            //id val point
            point_smoothness.push_back({(int)i, smoothness_val, cloud->points[i]});

        } else {
            // This case should ideally not happen if num_neighbors_ is reasonable and cloud is dense
            // Treat as an isolated/sharp point
            point_smoothness.push_back({(int)i, std::numeric_limits<float>::max(), cloud->points[i]});
        }
    }
    spdlog::info("Computed smoothness for {} points.", point_smoothness.size());
    return point_smoothness;
}

std::vector<int> FeatureExtractor::applyNMS(
    const std::vector<PointSmoothness>& sorted_points,
    float radius)
{
    std::vector<int> kept_indices;
    if (sorted_points.empty()) {
        return kept_indices;
    }

    // Use a boolean vector/map to keep track of points already taken (to avoid duplicates in output features)
    // We assume original point IDs are 0 to sorted_points.back().id for direct array access.
    // If IDs are sparse, use std::map<int, bool> is_point_taken.
    // For this example, let's use a map for robustness against sparse IDs.
    std::map<int, bool> is_original_id_taken;
    for(const auto& ps : sorted_points) {
        is_original_id_taken[ps.id] = false;
    }

    // Create a KdTree for efficient radius search among the sorted points' actual coordinates
    pcl::PointCloud<PointT>::Ptr temp_cloud_for_kdtree(new pcl::PointCloud<PointT>());
    temp_cloud_for_kdtree->reserve(sorted_points.size());
    for(const auto& ps : sorted_points) {
        temp_cloud_for_kdtree->push_back(ps.point);
    }
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(temp_cloud_for_kdtree); // KdTree built on the *sorted* points' spatial locations

    for (const auto& ps : sorted_points) {
        // Only consider this point if its original ID hasn't been "suppressed" by a better point already chosen
        if (!is_original_id_taken[ps.id]) {
            kept_indices.push_back(ps.id); // Add its original ID to the kept list
            is_original_id_taken[ps.id] = true; // Mark this point's original ID as taken

            // Find neighbors within 'radius' around the selected point 'ps.point'
            std::vector<int> neighborIdxSearch; // Indices in temp_cloud_for_kdtree
            std::vector<float> neighborSquaredDistance;

            kdtree.radiusSearch(ps.point, radius, neighborIdxSearch, neighborSquaredDistance);

            // Mark all *original IDs* of the neighbors within the radius as "taken"
            // (They are suppressed by the current point 'ps')
            for (int kdtree_idx_in_temp_cloud : neighborIdxSearch) {
                is_original_id_taken[sorted_points[kdtree_idx_in_temp_cloud].id] = true;
            }
        }
    }
    spdlog::info("Applied NMS, {} points kept with radius {:.4f}.", kept_indices.size(), radius);
    return kept_indices;
}


void FeatureExtractor::extractFeatures(
    const pcl::PointCloud<PointT>::Ptr& input_cloud,
    pcl::PointCloud<PointT>::Ptr& edge_features_out,
    pcl::PointCloud<PointT>::Ptr& surface_features_out)
{
    edge_features_out->clear();
    surface_features_out->clear();

    if (input_cloud->empty()) {
        spdlog::warn("Input cloud for feature extraction is empty.");
        return;
    }
    if (input_cloud->size() < num_neighbors_ + 1) {
        spdlog::warn("Input cloud size ({}) too small for feature extraction. Need at least {} points. Returning empty features.",
                     input_cloud->size(), num_neighbors_ + 1);
        return;
    }

    // 1. Compute smoothness for all points
    std::vector<PointSmoothness> all_points_smoothness = computeSmoothness(input_cloud);
    if (all_points_smoothness.empty()) {
        spdlog::error("Smoothness computation returned an empty list. No features extracted.");
        return;
    }

    // Sort points by smoothness in ascending order (smoothest first)
    std::sort(all_points_smoothness.begin(), all_points_smoothness.end());

    // 2. Select Edge Features
    // Edge points have high curvature (high smoothness value in our metric)
    std::vector<PointSmoothness> potential_edges;
    // Iterate from the end (rougher points) towards the beginning (smoother points)
    // as edge_threshold_ is a lower bound for 'roughness' for edge points.
    for (int i = all_points_smoothness.size() - 1; i >= 0; --i) {
        if (all_points_smoothness[i].smoothness > edge_threshold_) {
            potential_edges.push_back(all_points_smoothness[i]);
        } else {
            // Since sorted by smoothness (ascending), once we hit a point below threshold,
            // all subsequent points (earlier in the sorted vector) will also be below it.
            break;
        }
    }
    spdlog::info("Found {} potential edge points based on threshold.", potential_edges.size());

    // Apply NMS to potential edge points. Sorting ensures we pick the 'sharpest' first.
    // The `applyNMS` function processes in its input order (which is `potential_edges` sorted descending by smoothness).
    // So, we want to apply NMS on `potential_edges` as-is (descending by smoothness means sharpest first).
    std::vector<int> edge_indices_kept = applyNMS(potential_edges, NMS_radius_);
    for (int original_idx : edge_indices_kept) {
        // Find the point in the original input_cloud using its original_idx
        edge_features_out->push_back(input_cloud->points[original_idx]);
    }
    spdlog::info("Selected {} edge features after NMS.", edge_features_out->size());


    // 3. Select Surface Features
    // Surface points have low curvature (low smoothness value in our metric)
    std::vector<PointSmoothness> potential_surfaces;
    // Iterate from the beginning (smoothest points)
    for (const auto& ps : all_points_smoothness) {
        if (ps.smoothness < surface_threshold_) {
            potential_surfaces.push_back(ps);
        } else {
            // Once we hit a point above threshold, all subsequent points will also be above it.
            break;
        }
    }
    spdlog::info("Found {} potential surface points based on threshold.", potential_surfaces.size());

    // Apply NMS to potential surface points.
    // For surfaces, the `sorted_points` list (potential_surfaces) is already sorted from smoothest to roughest.
    // NMS will effectively pick the smoothest points in a region first.
    std::vector<int> surface_indices_kept = applyNMS(potential_surfaces, NMS_radius_ * 2.0f); // Often a larger NMS radius for surfaces
    for (int original_idx : surface_indices_kept) {
        // Find the point in the original input_cloud using its original_idx
        surface_features_out->push_back(input_cloud->points[original_idx]);
    }
    spdlog::info("Selected {} surface features after NMS.", surface_features_out->size());
}