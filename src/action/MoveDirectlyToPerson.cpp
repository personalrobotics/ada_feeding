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
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  const aikido::constraint::dart::CollisionFreePtr& collisionFree = feedingDemo->getCollisionConstraint();
  double horizontalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("horizontalTolerance");
  double verticalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("verticalTolerance");
  double planningTimeout = feedingDemo->mPlanningTimeout;
  int maxNumTrials = feedingDemo->mMaxNumTrials;
  const Eigen::Vector6d& velocityLimits = feedingDemo->mVelocityLimits;

  Eigen::Isometry3d person = Eigen::Isometry3d::Identity();
  person.translation() = personPose.translation();
  person.linear()
      = Eigen::Matrix3d(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  if (tiltOffset)
  {
    person.translation() += *tiltOffset;
  }

  TSR personTSR;
  personTSR.mT0_w = person;
  // TODO: Remove this Erroneous offset
  personTSR.mTw_e.translation() = Eigen::Vector3d{0, 0.2, 0.15};

  if (tiltOffset)
  {
    personTSR.mBw = createBwMatrixForTSR(
        horizontalToleranceForPerson,
        horizontalToleranceForPerson,
        verticalToleranceForPerson,
        0,
        M_PI / 4,
        M_PI / 4);
    Eigen::Isometry3d eeTransform
        = ada->getHand()->getEndEffectorTransform("person").get();
    eeTransform.linear()
        = eeTransform.linear()
          * Eigen::Matrix3d(
                Eigen::AngleAxisd(M_PI * -0.25, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(M_PI * 0.25, Eigen::Vector3d::UnitX()));
    personTSR.mTw_e.matrix() *= eeTransform.matrix();
  }
  else
  {
    personTSR.mBw = createBwMatrixForTSR(
        horizontalToleranceForPerson,
        horizontalToleranceForPerson,
        verticalToleranceForPerson,
        0,
        0,
        0);
    personTSR.mTw_e.matrix()
        *= ada->getHand()->getEndEffectorTransform("person")->matrix();
  }

  // if (feedingDemo)
  //{
  //  feedingDemo->getViewer()->addTSRMarker(personTSR);
  //  std::cout << "check person TSR" << std::endl;
  //  int n;
  //  std::cin >> n;
  //}

  if (!ada->moveArmToTSR(
          personTSR,
          collisionFree,
          planningTimeout,
          maxNumTrials,
          getConfigurationRanker(ada),
          velocityLimits))
  {
    ROS_WARN_STREAM("Execution failed");
    return false;
  }
  return true;
}
} // namespace action
} // namespace feeding
