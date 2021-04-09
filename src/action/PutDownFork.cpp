#include "feeding/action/PutDownFork.hpp"

#include "feeding/TargetItem.hpp"
#include "feeding/action/MoveAbove.hpp"
#include "feeding/action/MoveAboveForque.hpp"
#include "feeding/action/MoveAbovePlate.hpp"
#include "feeding/action/MoveInto.hpp"
#include "feeding/action/MoveOutOf.hpp"

namespace feeding {
namespace action {

void putDownFork(
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
  ada->closeHand();
  moveAboveForque(
      ada,
      collisionFree,
      forkHolderAngle,
      forkHolderTranslation,
      planningTimeout,
      maxNumTrials);

  moveInto(
      ada,
      nullptr,
      collisionFree,
      nullptr,
      TargetItem::FORQUE,
      planningTimeout,
      endEffectorOffsetPositionTolerance,
      endEffectorOffsetAngularTolerance,
      Eigen::Vector3d(0, 1, 0), // direction
      ftThresholdHelper);

  ada->openHand();
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  moveOutOf(
      ada,
      collisionFree,
      TargetItem::FORQUE,
      0.04,                      // length
      Eigen::Vector3d(0, -1, 0), // direction
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
