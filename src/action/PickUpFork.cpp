#include "feeding/action/PickUpFork.hpp"

#include "feeding/TargetItem.hpp"
#include "feeding/action/MoveAboveForque.hpp"
#include "feeding/action/MoveAbovePlate.hpp"
#include "feeding/action/MoveInto.hpp"
#include "feeding/action/MoveOutOf.hpp"
#include "feeding/util.hpp"

namespace feeding {
namespace action {

void pickUpFork(
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  const aikido::constraint::dart::CollisionFreePtr& collisionFree = feedingDemo->getCollisionConstraint();

  ada->openHand();
  moveAboveForque(
      collisionFree,
      feedingDemo);

  Eigen::Vector3d endEffectorDirection(0, 0, -1);
  moveInto(
      nullptr,
      TargetItem::FORQUE,
      endEffectorDirection,
      feedingDemo);

  std::vector<std::string> optionPrompts{"(1) close", "(2) leave-as-is"};
  auto input = getUserInputWithOptions(optionPrompts, "Close Hand?");

  if (input == 1)
  {
    ada->closeHand();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }

  double length = 0.04;
  Eigen::Vector3d direction(0, -1, 0);

  moveOutOf(
      nullptr, // ignore collision
      TargetItem::FORQUE,
      length,
      direction,
      feedingDemo);

  moveAbovePlate(
      plate,
      plateEndEffectorTransform,
      feedingDemo);
}

} // namespace action
} // namespace feeding
