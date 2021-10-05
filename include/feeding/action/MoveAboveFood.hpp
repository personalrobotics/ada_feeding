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
    std::string foodName,
    const Eigen::Isometry3d& foodTransform,
    float rotateAngle,
    TiltStyle tiltStyle,
    double rotationTolerance,
    FeedingDemo* feedingDemo,
    double* angleGuess = nullptr);

} // namespace action
} // namespace feeding

#endif
