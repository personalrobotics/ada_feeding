#ifndef FEEDING_ACTION_MOVEABOVEPLATE_HPP_
#define FEEDING_ACTION_MOVEABOVEPLATE_HPP_

#include <libada/Ada.hpp>

namespace feeding {
namespace action {

bool moveAbovePlate(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    double horizontalTolerance,
    double verticalTolerance,
    double rotationTolerance,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits);

} // namespace action
} // namespace feeding

#endif