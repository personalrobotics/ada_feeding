#ifndef FEEDING_ACTION_MOVEDIRECTLYTO_HPP_
#define FEEDING_ACTION_MOVEDIRECTLYTO_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"

namespace feeding {
namespace action {

bool moveDirectlyToPerson(
    const Eigen::Isometry3d& personPose,
    const Eigen::Vector3d* tiltOffset,
    FeedingDemo* feedingDemo);

} // namespace action
} // namespace feeding

#endif
