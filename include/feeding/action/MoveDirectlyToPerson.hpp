#ifndef FEEDING_ACTION_MOVEDIRECTLYTO_HPP_
#define FEEDING_ACTION_MOVEDIRECTLYTO_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"

namespace feeding {
namespace action {

bool moveDirectlyToPerson(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const Eigen::Isometry3d& personPose,
    double distanceToPerson,
    double horizontalToleranceForPerson,
    double verticalToleranceForPerson,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits,
    const Eigen::Vector3d* tiltOffset,
    FeedingDemo* feedingDemo = nullptr);

} // namespace action
} // namespace feeding

#endif
