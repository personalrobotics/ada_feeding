#include "feeding/action/MoveAbovePlate.hpp"

#include "feeding/action/MoveAbove.hpp"

namespace feeding {
namespace action {

bool moveAbovePlate(const Eigen::Isometry3d &plate,
                    const Eigen::Isometry3d &plateEndEffectorTransform,
                    FeedingDemo *feedingDemo) {
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada> &ada = feedingDemo->getAda();
  const aikido::constraint::dart::CollisionFreePtr &collisionFree =
      feedingDemo->getCollisionConstraint();
  double horizontalTolerance =
      feedingDemo->mPlateTSRParameters.at("horizontalTolerance");
  double verticalTolerance =
      feedingDemo->mPlateTSRParameters.at("verticalTolerance");
  // NOTE: Although rotationTolerance was originally passed in as a param, it
  // was never used. double rotationTolerance =
  // feedingDemo->mPlateTSRParameters.at("rotationTolerance");
  auto trajectory = ada->getArm()->planToConfiguration(
      ada->getArm()->getNamedConfiguration("home_config"),
      ada->getArm()->getWorldCollisionConstraint());
  bool success = true;
  auto future = ada->getArm()->executeTrajectory(
      trajectory); // check velocity limits are set in FeedingDemo
  try {
    future.get();
  } catch (const std::exception &e) {
    dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
    success = false;
  }
  return success;
}

} // namespace action
} // namespace feeding
