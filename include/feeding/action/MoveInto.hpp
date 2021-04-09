#ifndef FEEDING_ACTION_MOVEINTO_HPP_
#define FEEDING_ACTION_MOVEINTO_HPP_

#include <libada/Ada.hpp>

#include "feeding/TargetItem.hpp"
#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"

#include "feeding/FTThresholdHelper.hpp"

namespace feeding {
namespace action {

bool moveInto(
    const std::shared_ptr<ada::Ada>& ada,
    const std::shared_ptr<Perception>& perception,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const ::ros::NodeHandle* nodeHandle,
    TargetItem item,
    double planningTimeout,
    double endEffectorOffsetPositionTolerenace,
    double endEffectorOffsetAngularTolerance,
    const Eigen::Vector3d& endEffectorDirection,
    std::shared_ptr<FTThresholdHelper> ftThresholdHelper,
    const Eigen::Vector6d& velocityLimits = Eigen::Vector6d::Zero());

} // namespace action
} // namespace feeding
#endif