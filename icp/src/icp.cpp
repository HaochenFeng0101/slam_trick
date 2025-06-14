#include "icp.h"

ICPRegistration::ICPRegistration()
    : last_transformation_(Eigen::Matrix4f::Identity()),
      converged_(false),
      fitness_score_(std::numeric_limits<double>::max())
{
    // Default ICP parameters - these can be tuned
    icp_.setMaxCorrespondenceDistance(0.1); // 10 cm
    icp_.setTransformationEpsilon(1e-8);
    icp_.setMaximumIterations(100);
}

Eigen::Matrix4f ICPRegistration::align(const PointCloudXYZ_NoColor::Ptr& source_cloud,
                                     const PointCloudXYZ_NoColor::Ptr& target_cloud,
                                     const Eigen::Matrix4f& initial_guess)
{
    PointCloudXYZ_NoColor::Ptr aligned_cloud(new PointCloudXYZ_NoColor);

    icp_.setInputSource(source_cloud);
    icp_.setInputTarget(target_cloud);

    // Set initial guess if provided
    icp_.align(*aligned_cloud, initial_guess);

    converged_ = icp_.hasConverged();
    fitness_score_ = icp_.getFitnessScore();
    last_transformation_ = icp_.getFinalTransformation();

    if (converged_)
    {
        spdlog::info("ICP converged with a fitness score of: {}", fitness_score_);
    }
    else
    {
        spdlog::warn("ICP did not converge!");
    }

    return last_transformation_;
}

void ICPRegistration::setMaxCorrespondenceDistance(double dist)
{
    icp_.setMaxCorrespondenceDistance(dist);
}

void ICPRegistration::setTransformationEpsilon(double epsilon)
{
    icp_.setTransformationEpsilon(epsilon);
}

void ICPRegistration::setMaximumIterations(int iterations)
{
    icp_.setMaximumIterations(iterations);
}

Eigen::Matrix4f ICPRegistration::getLastTransformation() const
{
    return last_transformation_;
}

bool ICPRegistration::hasConverged() const
{
    return converged_;
}

double ICPRegistration::getFitnessScore() const
{
    return fitness_score_;
}