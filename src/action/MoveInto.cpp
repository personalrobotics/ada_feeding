#include "feeding/action/MoveInto.hpp"

#include <libada/util.hpp>

#include "feeding/TargetItem.hpp"
#include "feeding/action/MoveAbove.hpp"
#include "feeding/action/MoveOutOf.hpp"
#include "feeding/action/PutDownFork.hpp"
#include "feeding/perception/PerceptionServoClient.hpp"
#include "feeding/util.hpp"

namespace feeding {
namespace action {

bool moveInto(
    const std::shared_ptr<Perception>& perception,
    TargetItem item,
    const Eigen::Vector3d& endEffectorDirection,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  const ros::NodeHandle* nodeHandle = feedingDemo->getNodeHandle().get();
  const aikido::constraint::dart::CollisionFreePtr& collisionFree = feedingDemo->getCollisionConstraint();
  double planningTimeout = feedingDemo->mPlanningTimeout;
  double endEffectorOffsetPositionTolerance = feedingDemo->mEndEffectorOffsetPositionTolerance;
  double endEffectorOffsetAngularTolerance = feedingDemo->mEndEffectorOffsetAngularTolerance;
  std::shared_ptr<FTThresholdHelper> ftThresholdHelper = feedingDemo->getFTThresholdHelper();
  const Eigen::Vector6d& velocityLimits = feedingDemo->mVelocityLimits;

  ROS_INFO_STREAM("Move into " + TargetToString.at(item));

  if (item != FOOD && item != FORQUE)
    throw std::invalid_argument(
        "MoveInto[" + TargetToString.at(item) + "] not supported");

  if (item == TargetItem::FORQUE)
    return ada->moveArmToEndEffectorOffset(
        Eigen::Vector3d(0, 1, 0),
        0.01,
        collisionFree,
        planningTimeout,
        endEffectorOffsetPositionTolerance,
        endEffectorOffsetAngularTolerance,
        velocityLimits);

  // if (perception)
  // {
  //   ROS_INFO("Servoing into food");

  //   int numDofs = ada->getArm()->getMetaSkeleton()->getNumDofs();
  //   std::vector<double> velocityLimits(numDofs, 0.2);

  //   PerceptionServoClient servoClient(
  //       nodeHandle,
  //       boost::bind(&Perception::getTrackedFoodItemPose, perception.get()),
  //       ada->getArm()->getStateSpace(),
  //       ada,
  //       ada->getArm()->getMetaSkeleton(),
  //       ada->getHand()->getEndEffectorBodyNode(),
  //       ada->getTrajectoryExecutor(),
  //       nullptr,
  //       1.0,
  //       0.002,
  //       planningTimeout,
  //       endEffectorOffsetPositionTolerance,
  //       endEffectorOffsetAngularTolerance,
  //       true, // servoFood
  //       velocityLimits);
  //   servoClient.start();

  //   return servoClient.wait(15.0);
  // }
  // else

  std::cout << "endEffectorDirection " << endEffectorDirection.transpose()
            << std::endl;
  // int n;
  // std::cin >> n;
  {
    double length = 0.085;
    int numDofs = ada->getArm()->getMetaSkeleton()->getNumDofs();
    // Collision constraint is not set because f/t sensor stops execution.

    auto result = ada->moveArmToEndEffectorOffset(
        endEffectorDirection,
        length,
        nullptr,
        planningTimeout,
        endEffectorOffsetPositionTolerance,
        endEffectorOffsetAngularTolerance,
        velocityLimits);
    ROS_INFO_STREAM(" Execution result: " << result);
  }

  return true;
}

} // namespace action
} // namespace feeding
