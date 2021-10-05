#ifndef FEEDING_ACTION_FEEDFOODTOPERSON_HPP_
#define FEEDING_ACTION_FEEDFOODTOPERSON_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

void feedFoodToPerson(
    const std::shared_ptr<Perception>& perception,
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    const Eigen::Vector3d* tiltOffset,
    FeedingDemo* feedingDemo);
}
} // namespace feeding

#endif
