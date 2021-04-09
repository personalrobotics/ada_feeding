#ifndef FEEDING_ACTION_MOVEABOVEFORQUE_HPP_
#define FEEDING_ACTION_MOVEABOVEFORQUE_HPP_

#include <libada/Ada.hpp>

#include "feeding/Workspace.hpp"

namespace feeding {
namespace action {

void moveAboveForque(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    double forkHolderAngle,
    std::vector<double> forkHolderTranslation,
    double planningTimeout,
    int maxNumTrials);

} // namespace action
} // namespace feeding

#endif