#ifndef FEEDING_ACTION_MOVETOWARDSPERSON_HPP_
#define FEEDING_ACTION_MOVETOWARDSPERSON_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"

namespace feeding {
namespace action {

bool moveTowardsPerson(
    const aikido::constraint::dart::CollisionFreePtr &collisionFree,
    const std::shared_ptr<Perception> &perception, double distanceToPerson,
    FeedingDemo *feedingDemo);
}
} // namespace feeding

#endif
