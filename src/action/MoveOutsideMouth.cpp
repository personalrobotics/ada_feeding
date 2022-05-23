#include "feeding/action/MoveOutsideMouth.hpp"

#include <libada/util.hpp>

#include "feeding/perception/Perception.hpp"

using ada::util::createBwMatrixForTSR;
using aikido::constraint::dart::TSR;

namespace feeding {
namespace action {

bool moveOutsideMouth(
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    const std::shared_ptr<Perception>& perception,
    double distanceToPerson,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  const std::shared_ptr<::ada::Ada>& adaSim = feedingDemo->getAdaSimulation();
  const std::shared_ptr<Workspace>& workspace = feedingDemo->getWorkspace();
  const ros::NodeHandle* nodeHandle = feedingDemo->getNodeHandle().get();
  // const Eigen::Isometry3d& personPose = workspace->getPersonPose();
  double horizontalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("horizontalTolerance");
  double verticalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("verticalTolerance");
  double planningTimeout = feedingDemo->mPlanningTimeout;
  double endEffectorOffsetPositionTolerance = feedingDemo->mEndEffectorOffsetPositionTolerance;
  double endEffectorOffsetAngularTolerance = feedingDemo->mEndEffectorOffsetAngularTolerance;

  // Read Person Pose
  bool seePerson = false;
  Eigen::Isometry3d personPose;
  while (!seePerson)
  {
    try
    {
      personPose = perception->perceiveFace();
      seePerson = true;
    }
    catch (...)
    {
      ROS_WARN_STREAM("No Face Detected!");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
  }

  bool success = true;
  {
    ROS_INFO_STREAM("Tilt inside mouth");

    std::cout << "EE name : " << ada->getHand()->getEndEffectorBodyNode()->getName() << std::endl;

    Eigen::Isometry3d currentPose
        = ada->getHand()->getEndEffectorBodyNode()->getTransform();

    TSR personTSR;
    personTSR.mT0_w = currentPose;
    // TODO: Remove this Erroneous offset
    personTSR.mTw_e = Eigen::Isometry3d::Identity();
    personTSR.mTw_e.linear() = personTSR.mTw_e.linear()*Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX());

    personTSR.mBw = createBwMatrixForTSR(
        0.001,
        0.001,
        0.001,
        0,
        0,
        0);

    auto personTSRPtr = std::make_shared<aikido::constraint::dart::TSR>(personTSR);
    auto trajectory = ada->getArm()->planToTSR(
        ada->getEndEffectorBodyNode()->getName(),
        personTSRPtr, 
        ada->getArm()->getWorldCollisionConstraint(std::vector<std::string>{"plate", "table", "wheelchair","person"}));
    
    auto future = ada->getArm()->executeTrajectory(trajectory); // check velocity limits are set in FeedingDemo
    try
    {
      future.get();
    }
    catch (const std::exception& e)
    {
      dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
      success = false;
      return success;
    }
  }

  {

    ROS_INFO_STREAM("Move outside mouth");

    int numDofs = ada->getArm()->getMetaSkeleton()->getNumDofs();
    // FAST
    Eigen::Vector6d velocityLimits = Eigen::Vector6d::Ones() * 0.3;
    // SLOW
    // Eigen::Vector6d velocityLimits = Eigen::Vector6d::Ones() * 0.1;

    // Eigen::Isometry3d currentPose
    //     = ada->getHand()->getEndEffectorBodyNode()->getTransform();

    // // Plan from current to goal pose
    // Eigen::Vector3d vectorToGoalPose
    //     = personPose.translation() - currentPose.translation();
    // vectorToGoalPose.y() -= distanceToPerson;

    Eigen::Vector3d offsetPersonFrame = {0.06, 0.0, 0.00};

    Eigen::Vector3d offsetWorldFrame = personPose.linear() * offsetPersonFrame;

    // Eigen::Vector3d vectorToGoalPose = {0, -0.1, 0.02};
    auto length = offsetWorldFrame.norm();
    offsetWorldFrame.normalize();

    ROS_WARN_STREAM("Angular Tolerance: " << endEffectorOffsetAngularTolerance);
    ROS_WARN_STREAM("Pose Tolerance: " << endEffectorOffsetPositionTolerance);
    ROS_WARN_STREAM("Offset: " << distanceToPerson);
    ROS_WARN_STREAM("Goal Pose: " << offsetWorldFrame);

    auto trajectory = ada->getArm()->planToOffset(
        ada->getEndEffectorBodyNode()->getName(),
        offsetWorldFrame * length);

    try
    {
      auto sim_future = adaSim->getArm()->executeTrajectory(trajectory); // check velocity limits are set in FeedingDemo
      sim_future.get();
    }
    catch (const std::exception& e)
    {
      dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
    }


    std::cout<<"Executed simulation trajectory! Press [ENTER] to continue:";
    std::cin.get();std::cout<<"Press [ENTER] again: ";std::cin.get();


    // bool success = true;
    try
    {
      auto future = ada->getArm()->executeTrajectory(trajectory); // check velocity limits are set in FeedingDemo
      future.get();
    }
    catch (const std::exception& e)
    {
      dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
      success = false;
    }
  }
  return success;
}
} // namespace action
} // namespace feeding
