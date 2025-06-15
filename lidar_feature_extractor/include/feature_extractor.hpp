#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits> // For numeric_limits

#include <pcl/kdtree/kdtree_flann.h> // For neighborhood search
#include <Eigen/Dense> // For Eigenvalues

// Using PointXYZ for simplicity, but PointXYZI can be used if intensity is relevant
using PointT = pcl::PointXYZ;

// Structure to hold a point and its computed smoothness
struct PointSmoothness {
    int id; // Original index in the input cloud
    float smoothness; // The calculated smoothness/curvature value
    PointT point; // The point data itself

    // For sorting: operator< sorts in ascending order of smoothness
    bool operator<(const PointSmoothness& other) const {
        return smoothness < other.smoothness;
    }
};

class FeatureExtractor {
public:
    FeatureExtractor(); // Constructor to set default parameters

    /**
     * @brief Extracts edge and surface features from an input point cloud.
     * @param input_cloud The raw input point cloud.
     * @param edge_features_out Output cloud for edge features.
     * @param surface_features_out Output cloud for surface features.
     */
    void extractFeatures(
        const pcl::PointCloud<PointT>::Ptr& input_cloud,
        pcl::PointCloud<PointT>::Ptr& edge_features_out,
        pcl::PointCloud<PointT>::Ptr& surface_features_out);

private:
    int num_neighbors_;         // Number of neighbors to consider for smoothness calculation
    float edge_threshold_;      // Points with smoothness > edge_threshold are potential edges
    float surface_threshold_;   // Points with smoothness < surface_threshold are potential surfaces
    float NMS_radius_;          // Non-Maximum Suppression radius for feature thinning

    /**
     * @brief Computes the smoothness (curvature) for each point in a cloud.
     * Uses PCA on local neighborhood to determine planarity/linearity.
     * @param cloud The input cloud.
     * @return A vector of PointSmoothness structs, containing original IDs, smoothness values, and points.
     */
    std::vector<PointSmoothness> computeSmoothness(
        const pcl::PointCloud<PointT>::Ptr& cloud);

    /**
     * @brief Applies Non-Maximum Suppression (NMS) to thin out features.
     * It iterates through sorted points and keeps only the "best" point within a given radius,
     * marking others in that radius as "taken" (suppressed).
     * @param sorted_points A vector of PointSmoothness structs, already sorted by smoothness.
     * @param radius The radius for NMS (points within this distance will be suppressed).
     * @return A vector of original indices (IDs) of the points that are kept after NMS.
     */
    std::vector<int> applyNMS(
        const std::vector<PointSmoothness>& sorted_points,
        float radius);
};

#endif // FEATURE_EXTRACTOR_HPP