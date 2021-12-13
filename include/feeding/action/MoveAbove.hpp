#ifndef FEEDING_ACTION_MOVEABOVE_HPP_
#define FEEDING_ACTION_MOVEABOVE_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/Workspace.hpp"

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

bool moveAbove(const Eigen::Isometry3d &targetTransform,
               const Eigen::Isometry3d &endEffectorTransform,
               double horizontalTolerance, double verticalTolerance,
               double rotationTolerance, double tiltTolerance,
               FeedingDemo *feedingDemo);
}
} // namespace feeding

#endif
