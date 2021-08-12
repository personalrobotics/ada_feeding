#ifndef FEEDING_ACTION_MOVEABOVEFORQUE_HPP_
#define FEEDING_ACTION_MOVEABOVEFORQUE_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/Workspace.hpp"

namespace feeding {
namespace action {

void moveAboveForque(
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    FeedingDemo* feedingDemo);

} // namespace action
} // namespace feeding

#endif
