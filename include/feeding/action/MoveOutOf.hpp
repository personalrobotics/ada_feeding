#ifndef FEEDING_ACTION_MOVEOUTOF_HPP_
#define FEEDING_ACTION_MOVEOUTOF_HPP_

#include <libada/Ada.hpp>

#include "feeding/FTThresholdHelper.hpp"
#include "feeding/TargetItem.hpp"
#include "feeding/Workspace.hpp"

namespace feeding {
namespace action {

void moveOutOf(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    TargetItem item,
    double length,
    Eigen::Vector3d direction,
    double planningTimeout,
    double endEffectorOffsetPositionTolerance,
    double endEffectorOffsetAngularTolerance,
    const std::shared_ptr<FTThresholdHelper>& ftThresholdHelper,
    const Eigen::Vector6d& velocityLimits = Eigen::Vector6d::Zero());
}
} // namespace feeding

#endif