#ifndef FEEDING_ACTION_DETECTANDMOVEABOVEFOOD_HPP_
#define FEEDING_ACTION_DETECTANDMOVEABOVEFOOD_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/FoodItem.hpp"
#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

std::unique_ptr<FoodItem> detectAndMoveAboveFood(
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const std::shared_ptr<Perception>& perception,
    const std::string& foodName,
    double heightAboveFood,
    double horizontalTolerance,
    double verticalTolerance,
    double rotationTolerance,
    double tiltTolerance,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits,
    FeedingDemo* feedingDemo = nullptr,
    double* angleGuess = nullptr,
    int actionOverride = -1);
}
} // namespace feeding

#endif