#ifndef FEEDING_ACTION_MOVEABOVEFOOD_HPP_
#define FEEDING_ACTION_MOVEABOVEFOOD_HPP_

#include <libada/Ada.hpp>

#include "feeding/AcquisitionAction.hpp"
#include "feeding/FeedingDemo.hpp"
#include "feeding/Workspace.hpp"

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

bool moveAboveFood(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    std::string foodName,
    const Eigen::Isometry3d& foodTransform,
    float rotateAngle,
    TiltStyle tiltStyle,
    double heightAboveFood,
    double horizontalTolerance,
    double verticalTolerance,
    double rotationTolerance,
    double tiltTolerance,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits,
    FeedingDemo* feedingDemo = nullptr,
    double* angleGuess = nullptr);

} // namespace action
} // namespace feeding

#endif