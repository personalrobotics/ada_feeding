#include "feeding/action/MoveAbovePlate.hpp"

#include "feeding/action/MoveAbove.hpp"

namespace feeding {
namespace action {

bool moveAbovePlate(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    double horizontalTolerance,
    double verticalTolerance,
    double rotationTolerance,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits)
{

  // Hardcoded pose
  Eigen::VectorXd homeConfig(6);
  homeConfig << -2.11666, 3.34967, 2.04129, -2.30031, -2.34026, 2.9545;
  bool success = ada->moveArmToConfiguration(
      homeConfig, collisionFree, 2.0, velocityLimits);
  if (!success)
    return moveAbove(
        ada,
        collisionFree,
        plate,
        plateEndEffectorTransform,
        horizontalTolerance,
        verticalTolerance,
        M_PI,
        0.03,
        planningTimeout,
        maxNumTrials,
        velocityLimits);
  else
    return success;
}

} // namespace action
} // namespace feeding
