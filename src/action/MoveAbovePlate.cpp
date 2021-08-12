#include "feeding/action/MoveAbovePlate.hpp"

#include "feeding/action/MoveAbove.hpp"

namespace feeding {
namespace action {

bool moveAbovePlate(
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  const aikido::constraint::dart::CollisionFreePtr& collisionFree = feedingDemo->getCollisionConstraint();
  double horizontalTolerance = feedingDemo->mPlateTSRParameters.at("horizontalTolerance");
  double verticalTolerance = feedingDemo->mPlateTSRParameters.at("verticalTolerance");
  // NOTE: Although rotationTolerance was originally passed in as a param, it was never used.
  // double rotationTolerance = feedingDemo->mPlateTSRParameters.at("rotationTolerance");
  const Eigen::Vector6d& velocityLimits = feedingDemo->mVelocityLimits;

  // Hardcoded pose
  Eigen::VectorXd homeConfig(6);
  homeConfig << -2.11666, 3.34967, 2.04129, -2.30031, -2.34026, 2.9545;
  bool success = ada->moveArmToConfiguration(
      homeConfig, collisionFree, 2.0, velocityLimits);
  if (!success)
  {
    auto retval = moveAbove(
        plate,
        plateEndEffectorTransform,
        horizontalTolerance,
        verticalTolerance,
        M_PI,
        0.03,
        feedingDemo);
    return retval;
  }
  else
  {
    return success;
  }
}

} // namespace action
} // namespace feeding
