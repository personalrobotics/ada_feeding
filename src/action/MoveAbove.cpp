#include "feeding/action/MoveAbove.hpp"

#include <libada/util.hpp>

#include "feeding/util.hpp"
using ada::util::createBwMatrixForTSR;
using aikido::constraint::dart::CollisionFreePtr;
using aikido::constraint::dart::TSR;

// Contains motions which are mainly TSR actions
namespace feeding {

namespace action {

bool moveAbove(
    const std::shared_ptr<::ada::Ada>& ada,
    const CollisionFreePtr& collisionFree,
    const Eigen::Isometry3d& targetTransform,
    const Eigen::Isometry3d& endEffectorTransform,
    double horizontalTolerance,
    double verticalTolerance,
    double rotationTolerance,
    double tiltTolerance,
    double planningTimeout,
    int maxNumTrials,
    const Eigen::Vector6d& velocityLimits,
    FeedingDemo* feedingDemo)
{
  ROS_WARN_STREAM("CALLED MOVE ABOVE; Rotation: " << rotationTolerance);
  TSR target;

  target.mT0_w = targetTransform;
  target.mBw = createBwMatrixForTSR(
      horizontalTolerance,
      horizontalTolerance,
      verticalTolerance,
      0,
      tiltTolerance,
      rotationTolerance);

  target.mTw_e.matrix() = endEffectorTransform.matrix();

  try
  {
    bool trajectoryCompleted = false;
    do
    {
      std::cout << "MoveAbove Current pose \n"
                << ada->getMetaSkeleton()->getPositions().transpose()
                << std::endl;
      trajectoryCompleted = ada->moveArmToTSR(
          target,
          collisionFree,
          planningTimeout,
          maxNumTrials,
          getConfigurationRanker(ada),
          velocityLimits);

      if (!trajectoryCompleted)
      {
        if (rotationTolerance <= 2.0)
        {
          rotationTolerance *= 4;
          std::cout << "Trying again with rotation Tolerance:"
                    << rotationTolerance << std::endl;
          target.mBw = createBwMatrixForTSR(
              horizontalTolerance,
              horizontalTolerance,
              verticalTolerance,
              0,
              tiltTolerance,
              rotationTolerance);
          continue;
        }
      }
      else
      {
        break;
      }

    } while (rotationTolerance <= 2.0);
    if (!trajectoryCompleted)
    {
      // talk("No trajectory, check T.S.R.", true);
      if (feedingDemo && feedingDemo->getViewer())
      {
        feedingDemo->getViewer()->addTSRMarker(target);
        std::cout << "Check TSR" << std::endl;
        int n;
        std::cin >> n;
      }
    }
    return trajectoryCompleted;
  }
  catch (...)
  {
    ROS_WARN("Error in trajectory completion!");
    return false;
  }
}

} // namespace action
} // namespace feeding
