#ifndef FEEDING_ACTION_MOVEINFRONTOFPERSON_HPP_
#define FEEDING_ACTION_MOVEINFRONTOFPERSON_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/Workspace.hpp"

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

bool moveInFrontOfPerson(
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const Eigen::Isometry3d& workspacePersonPose,
    double distanceToPerson,
    double horizontalToleranceForPerson,
    double verticalToleranceForPerson,
    FeedingDemo* feedingDemo);
}
} // namespace feeding

#endif
