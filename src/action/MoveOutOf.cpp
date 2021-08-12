#include "feeding/action/MoveOutOf.hpp"

#include <libada/util.hpp>

#include "feeding/action/MoveAbove.hpp"
#include "feeding/action/MoveInto.hpp"
#include "feeding/util.hpp"

namespace feeding {
namespace action {

const static std::vector<std::string> trajectoryController{
    "rewd_trajectory_controller"};
const static std::vector<std::string> ftTrajectoryController{
    "move_until_touch_topic_controller"};

void moveOutOf(
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    TargetItem item,
    double length,
    Eigen::Vector3d direction,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  double planningTimeout = feedingDemo->mPlanningTimeout;
  double endEffectorOffsetPositionTolerance = feedingDemo->mEndEffectorOffsetPositionTolerance;
  double endEffectorOffsetAngularTolerance = feedingDemo->mEndEffectorOffsetAngularTolerance;
  std::shared_ptr<FTThresholdHelper> ftThresholdHelper = feedingDemo->getFTThresholdHelper();
  const Eigen::Vector6d& velocityLimits = feedingDemo->mVelocityLimits;

  ROS_INFO_STREAM("Move Out of " + TargetToString.at(item));

  if (ftThresholdHelper)
  {
    ftThresholdHelper->setThresholds(AFTER_GRAB_FOOD_FT_THRESHOLD);
  }

  bool trajectoryCompleted = ada->moveArmToEndEffectorOffset(
      direction,
      length,
      collisionFree,
      planningTimeout,
      endEffectorOffsetPositionTolerance,
      endEffectorOffsetAngularTolerance,
      velocityLimits);

  // trajectoryCompleted might be false because the forque hit the food
  // along the way and the trajectory was aborted
  if (ftThresholdHelper)
  {
    ftThresholdHelper->setThresholds(STANDARD_FT_THRESHOLD);
  }
}

} // namespace action
} // namespace feeding
