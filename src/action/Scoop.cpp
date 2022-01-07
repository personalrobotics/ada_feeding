#include "feeding/TargetItem.hpp"
#include "feeding/action/MoveAbove.hpp"
#include "feeding/action/MoveInto.hpp"
#include "feeding/action/PickUpFork.hpp"
#include "feeding/util/util.hpp"

namespace feeding {
namespace action {

// bool moveWithEndEffectorTwist(
//     const Eigen::Vector6d& twists,
//     double duration,
//     bool respectCollision)
// {
//    temporarily disabling
//   return mAda->moveArmWithEndEffectorTwist(
//     Eigen::Vector6d(getRosParam<std::vector<double>>("/scoop/twist1",
//     mNodeHandle).data()),
//     respectCollision ? mCollisionFreeConstraint : nullptr,
//     duration,
//     getRosParam<double>("/planning/timeoutSeconds", mNodeHandle),
//     getRosParam<double>(
//           "/planning/endEffectorTwist/positionTolerance", mNodeHandle),
//     getRosParam<double>(
//           "/planning/endEffectorTwist/angularTolerance", mNodeHandle));

// }

void scoop(const std::shared_ptr<ada::Ada> &ada) {
  //  temporarily disabling
  // std::vector<std::string> twists{
  //   "/scoop/twist1", "/scoop/twist2", "/scoop/twist3"};

  // for (const auto & param : twists)
  // {
  //   auto success =
  //     ada->moveWithEndEffectorTwist(
  //       Eigen::Vector6d(getRosParam<std::vector<double>>(param,
  //       mNodeHandle).data()));
  //   if (!success)
  //   {
  //     ROS_ERROR_STREAM("Failed to execute " << param << std::endl);
  //     throw std::runtime_error("Failed to execute scoop");
  //   }
  // }
}

} // namespace action
} // namespace feeding
