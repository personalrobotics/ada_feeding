#ifndef FEEDING_ACTION_MOVEOUTOF_HPP_
#define FEEDING_ACTION_MOVEOUTOF_HPP_

#include <libada/Ada.hpp>

#include "feeding/FTThresholdHelper.hpp"
#include "feeding/FeedingDemo.hpp"
#include "feeding/TargetItem.hpp"
#include "feeding/Workspace.hpp"

namespace feeding {
namespace action {

void moveOutOf(const aikido::constraint::dart::CollisionFreePtr &collisionFree,
               TargetItem item, double length, Eigen::Vector3d direction,
               FeedingDemo *feedingDemo);
}
} // namespace feeding

#endif
