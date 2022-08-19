#include "feeding/action/PutDownFork.hpp"

#include "feeding/TargetItem.hpp"
#include "feeding/action/MoveAbove.hpp"
#include "feeding/action/MoveAboveForque.hpp"
#include "feeding/action/MoveAbovePlate.hpp"
#include "feeding/action/MoveInto.hpp"
#include "feeding/action/MoveOutOf.hpp"

namespace feeding {
namespace action {

void putDownFork(const Eigen::Isometry3d &plate,
                 const Eigen::Isometry3d &plateEndEffectorTransform,
                 FeedingDemo *feedingDemo) {
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada> &ada = feedingDemo->getAda();
  const aikido::constraint::dart::CollisionFreePtr &collisionFree =
      feedingDemo->getCollisionConstraint();

  ada->closeHand();
  moveAboveForque(collisionFree, feedingDemo);

  moveInto(nullptr, TargetItem::FORQUE, Eigen::Vector3d(0, 1, 0), // direction
           feedingDemo);

  ada->openHand();
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  moveOutOf(collisionFree, TargetItem::FORQUE,
            0.04,                      // length
            Eigen::Vector3d(0, -1, 0), // direction
            feedingDemo);

  moveAbovePlate(plate, plateEndEffectorTransform, feedingDemo);
}

} // namespace action
} // namespace feeding
