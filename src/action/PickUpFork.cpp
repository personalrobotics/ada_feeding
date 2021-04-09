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
    const std::shared_ptr<ada::Ada>& ada,
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    double forkHolderAngle,
    std::vector<double> forkHolderTranslation,
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    double heightAbovePlate,
    double horizontalToleranceAbovePlate,
    double verticalToleranceAbovePlate,
    double rotationToleranceAbovePlate,
    double endEffectorOffsetPositionTolerance,
    double endEffectorOffsetAngularTolerance,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits,
    std::shared_ptr<FTThresholdHelper> ftThresholdHelper)
{
  ada->openHand();
  moveAboveForque(
      ada,
      collisionFree,
      forkHolderAngle,
      forkHolderTranslation,
      planningTimeout,
      maxNumTrials);

  Eigen::Vector3d endEffectorDirection(0, 0, -1);
  moveInto(
      ada,
      nullptr,
      collisionFree,
      nullptr,
      TargetItem::FORQUE,
      planningTimeout,
      endEffectorOffsetPositionTolerance,
      endEffectorOffsetAngularTolerance,
      endEffectorDirection,
      ftThresholdHelper);

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
      ada,
      nullptr, // ignore collision
      TargetItem::FORQUE,
      length,
      direction,
      planningTimeout,
      endEffectorOffsetPositionTolerance,
      endEffectorOffsetAngularTolerance,
      ftThresholdHelper);

  moveAbovePlate(
      ada,
      collisionFree,
      plate,
      plateEndEffectorTransform,
      horizontalToleranceAbovePlate,
      verticalToleranceAbovePlate,
      rotationToleranceAbovePlate,
      planningTimeout,
      maxNumTrials,
      velocityLimits);
}

} // namespace action
} // namespace feeding
