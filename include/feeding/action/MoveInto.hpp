#ifndef FEEDING_ACTION_MOVEINTO_HPP_
#define FEEDING_ACTION_MOVEINTO_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/TargetItem.hpp"
#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"

#include "feeding/FTThresholdHelper.hpp"

namespace feeding {
namespace action {

bool moveInto(
    const std::shared_ptr<Perception>& perception,
    const ::ros::NodeHandle* nodeHandle,
    TargetItem item,
    const Eigen::Vector3d& endEffectorDirection,
    FeedingDemo* feedingDemo);

} // namespace action
} // namespace feeding
#endif
