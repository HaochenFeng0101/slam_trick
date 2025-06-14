#ifndef ICP_H
#define ICP_H

#include <ros/ros.h> // Still need ROS for Eigen types and some ROS/PCL utilities
#include <pcl_ros/point_cloud.h> // For PCL ROS utilities (e.g., fromROSMsg if you keep that part)
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
// #include <tf/transform_broadcaster.h> // No longer directly used by ICP class
// #include <tf/transform_listener.h>   // No longer directly used by ICP class
#include <spdlog/spdlog.h> // Include spdlog for logging within ICP class

// Define PointCloud types
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ_NoColor;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB_Color; // For visualization/saving

class ICPRegistration
{
public:
    ICPRegistration();

    /**
     * @brief Performs ICP registration between source and target point clouds.
     * @param source_cloud The source point cloud (will be transformed).
     * @param target_cloud The target point cloud (fixed).
     * @param initial_guess The initial transformation guess (optional).
     * @return The resulting transformation matrix.
     */
    Eigen::Matrix4f align(const PointCloudXYZ_NoColor::Ptr& source_cloud,
                          const PointCloudXYZ_NoColor::Ptr& target_cloud,
                          const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity());

    /**
     * @brief Sets the maximum correspondence distance for ICP.
     * @param dist The maximum distance.
     */
    void setMaxCorrespondenceDistance(double dist);

    /**
     * @brief Sets the transformation epsilon for ICP.
     * @param epsilon The transformation epsilon.
     */
    void setTransformationEpsilon(double epsilon);

    /**
     * @brief Sets the maximum number of ICP iterations.
     * @param iterations The maximum iterations.
     */
    void setMaximumIterations(int iterations);

    /**
     * @brief Gets the last used ICP transformation matrix.
     * @return The last transformation matrix.
     */
    Eigen::Matrix4f getLastTransformation() const;

    /**
     * @brief Checks if the ICP convergence was successful.
     * @return True if converged, false otherwise.
     */
    bool hasConverged() const;

    /**
     * @brief Returns the fitness score of the ICP registration.
     * @return The fitness score.
     */
    double getFitnessScore() const;


private:
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;
    Eigen::Matrix4f last_transformation_;
    bool converged_;
    double fitness_score_;
};

#endif // ICP_H