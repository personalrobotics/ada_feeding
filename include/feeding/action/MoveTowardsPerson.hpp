#ifndef FEEDING_ACTION_MOVETOWARDSPERSON_HPP_
#define FEEDING_ACTION_MOVETOWARDSPERSON_HPP_

#include <libada/Ada.hpp>

#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"

namespace feeding {
namespace action {

bool moveTowardsPerson(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const std::shared_ptr<Perception>& perception,
    const ros::NodeHandle* nodeHandle,
    double distanceToPerson,
    double planningTimeout,
    double endEffectorOffsetPositionTolerenace,
    double endEffectorOffsetAngularTolerance);
}
} // namespace feeding

#endif
