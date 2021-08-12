#include "feeding/action/MoveInFrontOfPerson.hpp"

#include <libada/util.hpp>

#include "feeding/util.hpp"

using ada::util::createBwMatrixForTSR;
using aikido::constraint::dart::TSR;

using aikido::constraint::dart::TSR;

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

bool moveInFrontOfPerson(
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const Eigen::Isometry3d& workspacePersonPose,
    double distanceToPerson,
    double horizontalToleranceForPerson,
    double verticalToleranceForPerson,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  double planningTimeout = feedingDemo->mPlanningTimeout;
  int maxNumTrials = feedingDemo->mMaxNumTrials;
  const Eigen::Vector6d& velocityLimits = feedingDemo->mVelocityLimits;

  ROS_INFO_STREAM("move in front of person");

  // hardcoded pose in front of person
  Eigen::VectorXd moveIFOPose(6);
  // Wheelchair
  moveIFOPose << -2.30252, 4.23221, 3.84109, -4.65546, 3.94225, 4.26543;

  // Tripod
  // moveIFOPose << -1.81753, 4.32404, 4.295815, 3.12878, 1.89724, -0.61526;

  // Participant
  // moveIFOPose << -2.30293, 4.04904, 3.63059, 1.62787, -2.34089, -2.01773;

  // Participant Tripod
  // moveIFOPose << -1.81752, 4.60286, 4.64300, -3.05122, 1.89743, -0.61493;

  bool success = ada->moveArmToConfiguration(
      moveIFOPose, collisionFree, 2.0, velocityLimits);
  if (success)
    return true;

  TSR personTSR;
  Eigen::Isometry3d personPose = Eigen::Isometry3d::Identity();
  personPose.translation() = workspacePersonPose.translation();
  personPose.linear()
      = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  personTSR.mT0_w = personPose;
  personTSR.mTw_e.translation() = Eigen::Vector3d{0, distanceToPerson, 0};

  personTSR.mBw = createBwMatrixForTSR(
      horizontalToleranceForPerson,
      horizontalToleranceForPerson,
      verticalToleranceForPerson,
      0,
      0,
      0);
  personTSR.mTw_e.matrix()
      *= ada->getHand()->getEndEffectorTransform("person")->matrix();

  return ada->moveArmToTSR(
      personTSR,
      collisionFree,
      planningTimeout,
      maxNumTrials,
      getConfigurationRanker(ada),
      velocityLimits);
}
} // namespace action
} // namespace feeding
