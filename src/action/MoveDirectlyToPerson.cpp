#include "feeding/action/MoveDirectlyToPerson.hpp"

#include <libada/util.hpp>

#include "feeding/util.hpp"

using ada::util::createBwMatrixForTSR;
using aikido::constraint::dart::TSR;

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

// TODO: This gets to MoveInfrontOfPerson pose. need moveInto?
bool moveDirectlyToPerson(
    const Eigen::Isometry3d& personPose,
    const Eigen::Vector3d* tiltOffset,
    FeedingDemo* feedingDemo)
{

  std::cout<<"IN moveDirectlyToPerson!"<<std::endl<<std::endl;
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  const aikido::constraint::dart::CollisionFreePtr& collisionFree = feedingDemo->getCollisionConstraint();
  double horizontalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("horizontalTolerance");
  double verticalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("verticalTolerance");
  double planningTimeout = feedingDemo->mPlanningTimeout;
  int maxNumTrials = feedingDemo->mMaxNumTrials;
  const Eigen::Vector6d& velocityLimits = feedingDemo->mVelocityLimits;

  Eigen::Matrix3d m = personPose.linear();
  std::cout << "personPose rotation: " << std::endl << m << std::endl;

  Eigen::Vector3d v = personPose.translation();
  std::cout << "personPose translation: " << std::endl << v << std::endl;

  TSR personTSR;
  personTSR.mT0_w = personPose;

  Eigen::Isometry3d eeTransformForque;
  eeTransformForque = Eigen::Isometry3d::Identity();
  eeTransformForque.linear() = eeTransformForque.linear()
      *Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitY())
      *Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ());
  eeTransformForque.translation() = Eigen::Vector3d{0.06, 0, 0};

  personTSR.mTw_e = eeTransformForque;

  personTSR.mBw = createBwMatrixForTSR(
      horizontalToleranceForPerson,
      horizontalToleranceForPerson,
      verticalToleranceForPerson,
      0,
      0,
      0);

  // if (feedingDemo)
  //{
  //  feedingDemo->getViewer()->addTSRMarker(personTSR);
  //  std::cout << "check person TSR" << std::endl;
  //  int n;
  //  std::cin >> n;
  //}

  std::cout<<"Frame used for planToTSR: "<<ada->getEndEffectorBodyNode()->getName()<<std::endl;
  auto personTSRPtr = std::make_shared<aikido::constraint::dart::TSR>(personTSR);
  auto trajectory = ada->getArm()->planToTSR(
      ada->getEndEffectorBodyNode()->getName(),
      personTSRPtr, 
      ada->getArm()->getWorldCollisionConstraint());
  bool success = true;
  auto future = ada->getArm()->executeTrajectory(trajectory); // check velocity limits are set in FeedingDemo
  try
  {
    future.get();
  }
  catch (const std::exception& e)
  {
    dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
    success = false;
  }

  if(!success)
  {
    ROS_WARN_STREAM("Execution failed");
    return false;
  }
  return true;
}
} // namespace action
} // namespace feeding
