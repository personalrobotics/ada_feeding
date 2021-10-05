#ifndef FEEDING_ACTION_PICKUPFORK_HPP_
#define FEEDING_ACTION_PICKUPFORK_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/FTThresholdHelper.hpp"

namespace feeding {
namespace action {

void pickUpFork(
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    FeedingDemo* feedingDemo);
}
} // namespace feeding

#endif
