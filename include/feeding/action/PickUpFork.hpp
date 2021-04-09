#ifndef FEEDING_ACTION_PICKUPFORK_HPP_
#define FEEDING_ACTION_PICKUPFORK_HPP_

#include <libada/Ada.hpp>

#include "feeding/FTThresholdHelper.hpp"

namespace feeding {
namespace action {

void pickUpFork(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    double forkHolderAngle,
    std::vector<double> forkHolderTranslation,
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    double heightAbovePlate,
    double horizontalToleranceAbovePlate,
    double verticalToleranceAbovePlate,
    double rotationToleranceAbovePlate,
    double endEffectorOffsetPositionTolerance,
    double endEffectorOffsetAngularTolerance,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits,
    std::shared_ptr<FTThresholdHelper> ftThresholdHelper);
}
} // namespace feeding

#endif
