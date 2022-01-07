#include "feeding/action/MoveInFrontOfPerson.hpp"

#include <libada/util.hpp>

#include "feeding/util.hpp"

using ada::util::createBwMatrixForTSR;
using aikido::constraint::dart::TSR;

using aikido::constraint::dart::TSR;

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

bool moveInFrontOfPerson(const aikido::constraint::TestablePtr &collisionFree,
                         const Eigen::Isometry3d &workspacePersonPose,
                         double distanceToPerson,
                         double horizontalToleranceForPerson,
                         double verticalToleranceForPerson,
                         FeedingDemo *feedingDemo) {
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada> &ada = feedingDemo->getAda();
  double planningTimeout = feedingDemo->mPlanningTimeout;
  int maxNumTrials = feedingDemo->mMaxNumTrials;
  int batchSize = feedingDemo->mBatchSize;
  int maxNumBatches = feedingDemo->mMaxNumBatches;
  int numMaxIterations = feedingDemo->mNumMaxIterations;
  const Eigen::Vector6d &velocityLimits = feedingDemo->mVelocityLimits;

  ROS_INFO_STREAM("move in front of person");

  auto trajectory = ada->getArm()->planToConfiguration(
      ada->getArm()->getNamedConfiguration("in_front_person_pose"),
      collisionFree);
  bool success = true;
  try {
    auto future = ada->getArm()->executeTrajectory(
        trajectory); // check velocity limits are set in FeedingDemo
    future.get();
  } catch (const std::exception &e) {
    dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
    success = false;
  }
  if (success)
    return true;

  else
    return false;
  /*

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

  // TODO: Remove hardcoded transform for person
  Eigen::Isometry3d eeTransformPerson;
  Eigen::Matrix3d rot;
  rot << 1., 0., 0.,
          0., 0., -1,
          0., 1., 0.;
  eeTransformPerson.linear() = rot;
  personTSR.mTw_e.matrix()
      *= eeTransformPerson.matrix();

  auto personTSRPtr =
  std::make_shared<aikido::constraint::dart::TSR>(personTSR); auto
  tsr_trajectory = ada->getArm()->planToTSR(
      ada->getEndEffectorBodyNode()->getName(),
      personTSRPtr,
      collisionFree,
      aikido::robot::util::PlanToTSRParameters(
        maxNumTrials,
        batchSize,
        maxNumBatches,
        numMaxIterations));
  bool tsr_success = true;
  auto tsr_future = ada->getArm()->executeTrajectory(tsr_trajectory); // check
  velocity limits are set in FeedingDemo try
  {
    tsr_future.get();
  }
  catch (const std::exception& e)
  {
    dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
    tsr_success = false;
  }

  return tsr_success;
  */
}
} // namespace action
} // namespace feeding
