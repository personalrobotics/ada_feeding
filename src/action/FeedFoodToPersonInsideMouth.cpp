#include "feeding/action/FeedFoodToPersonInsideMouth.hpp"

#include <libada/util.hpp>

#include "feeding/action/Grab.hpp"
#include "feeding/action/MoveAbovePlate.hpp"
#include "feeding/action/MoveDirectlyToPerson.hpp"
#include "feeding/action/MoveInFrontOfPerson.hpp"
#include "feeding/action/MoveInsideMouth.hpp"
#include "feeding/action/MoveOutsideMouth.hpp"
#include "feeding/util.hpp"

using ada::util::createBwMatrixForTSR;
using ada::util::getRosParam;
using aikido::constraint::dart::TSR;

namespace feeding {
namespace action {

static const std::vector<std::string> optionPrompts{"(1) tilt", "(2) no tilt"};
//==============================================================================
void feedFoodToPersonInsideMouth(
    const std::shared_ptr<Perception>& perception,
    const Eigen::Isometry3d& plate,
    const Eigen::Isometry3d& plateEndEffectorTransform,
    const Eigen::Vector3d* tiltOffset,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  const std::shared_ptr<Workspace>& workspace = feedingDemo->getWorkspace();
  const ros::NodeHandle* nodeHandle = feedingDemo->getNodeHandle().get();
  const aikido::constraint::dart::CollisionFreePtr& collisionFree = feedingDemo->getCollisionConstraint();
  // const aikido::constraint::dart::CollisionFreePtr& collisionFreeWithWallFurtherBack = feedingDemo->getCollisionConstraintWithWallFurtherBack();
  const Eigen::Isometry3d& personPose = workspace->getPersonPose();
  std::chrono::milliseconds waitAtPerson = feedingDemo->mWaitTimeForPerson;
  double distanceToPerson = feedingDemo->mPersonTSRParameters.at("distance");
  double horizontalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("horizontalTolerance");
  double verticalToleranceForPerson = feedingDemo->mPersonTSRParameters.at("verticalTolerance");
  double planningTimeout = feedingDemo->mPlanningTimeout;
  int maxNumTrials = feedingDemo->mMaxNumTrials;
  const Eigen::Vector6d& velocityLimits = feedingDemo->mVelocityLimits;

  // auto moveIFOPerson = [&] {
  //   auto retval = moveInFrontOfPerson(
  //       ada->getArm()->getWorldCollisionConstraint(),
  //       personPose,
  //       distanceToPerson,
  //       horizontalToleranceForPerson,
  //       verticalToleranceForPerson,
  //       feedingDemo);
  //   return retval;
  // }; 


  {
      // Read Person Pose
    bool seePerson = false;
    Eigen::Isometry3d personPose;
    while (!seePerson) {
      try {
        std::cout<<"In seePerson!"<<std::endl;
        personPose = perception->perceiveFace();
        // seePerson = true;
      } catch (...) {
        ROS_WARN_STREAM("No Face Detected!");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
    }
  }


  std::cout<<"Press [ENTER] to moveInFrontOfPerson:"<<std::endl;
  std::cin.get();std::cout<<"Press [ENTER] again: ";std::cin.get();

  auto moveIFOPerson = [&] {
    auto retval = moveInFrontOfPerson(
        ada->getArm()->getWorldCollisionConstraint(),
        personPose,
        distanceToPerson,
        horizontalToleranceForPerson,
        verticalToleranceForPerson,
        feedingDemo);
    return retval;
  };

  bool moveIFOSuccess = false;
  bool moveSuccess = false;
  for (std::size_t i = 0; i < 2; ++i)
  {
    moveIFOSuccess = moveIFOPerson();
    if (!moveIFOSuccess)
    {
      ROS_WARN_STREAM("Failed to move in front of person, retry");
      talk("Sorry, I'm having a little trouble moving. Let me try again.");
      continue;
    }
    else
      break;
  }

  // moveIFOSuccess = true;

  // std::cout<<"Press [ENTER] to switch controllers:"<<std::endl;
  // std::cin.get();std::cout<<"Press [ENTER] again: ";std::cin.get();

  ada->switchControllersHack(std::string("EFFORT"),
    std::string("towards_mouth_bite_transfer_controller"),std::string("move_until_touch_topic_controller"));

  std::cout<<"Successfully switched controllers!"<<std::endl;

  ada->switchTrajectoryLimits(std::string("transfer"));

  std::cout<<"Successfully switched trajectory limits!"<<std::endl;

  // Send message to web interface to indicate skewer finished
  publishActionDoneToWeb((ros::NodeHandle*)nodeHandle);

  // Ask for Tilt Override
  auto overrideTiltOffset = tiltOffset;

  publishTransferDoneToWeb((ros::NodeHandle*)nodeHandle);

  if (moveIFOSuccess)
  {

    nodeHandle->setParam("/feeding/facePerceptionOn", true);

    // Read Person Pose
    bool seePerson = false;
    Eigen::Isometry3d personPose;
    while (!seePerson) {
      try {
        std::cout<<"In seePerson!"<<std::endl;
        personPose = perception->perceiveFace();
        seePerson = true;
      } catch (...) {
        ROS_WARN_STREAM("No Face Detected!");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
    }

    // std::cout<<"Press [ENTER] to moveDirectlyToPerson:"<<std::endl;
    // std::cin.get();std::cout<<"Press [ENTER] again: ";std::cin.get();


    ROS_INFO_STREAM("Move infront of mouth");
    moveSuccess = moveDirectlyToPerson(
        personPose,
        nullptr,
        feedingDemo);

    // std::cout<<"ABORT CODE:"<<std::endl;
    // std::cin.get();std::cout<<"REPEAT! ABORT CODE: ";std::cin.get();

    moveSuccess = moveDirectlyToPerson(
        personPose,
        nullptr,
        feedingDemo);

    if(moveSuccess)
    {
      if (overrideTiltOffset == nullptr)
      {
        distanceToPerson = 0;
      }

      ada->switchControllersHack(std::string(""),
      std::string("inside_mouth_bite_transfer_controller"),std::string("towards_mouth_bite_transfer_controller"));

      ROS_INFO_STREAM("Move towards person");

      std::cout<<"Press [ENTER] to moveInsideMouth:"<<std::endl;
      std::cin.get();std::cout<<"Press [ENTER] again: ";std::cin.get();

      moveSuccess = moveInsideMouth(
          nullptr,
          perception,
          distanceToPerson,
          feedingDemo);

      // std::cout<<"ABORT CODE:"<<std::endl;
      // std::cin.get();std::cout<<"REPEAT! ABORT CODE: ";std::cin.get();
    }
    nodeHandle->setParam("/feeding/facePerceptionOn", false);
  }

  if (moveIFOSuccess)
  {
    // ===== EATING =====
    ROS_WARN("Human is eating");
    talk("Ready to eat!");
    std::this_thread::sleep_for(waitAtPerson);

    // Backward
    ada::util::waitForUser("Move backward", ada);
    talk("Let me get out of your way.", true);

    moveSuccess = moveOutsideMouth(
          nullptr,
          perception,
          distanceToPerson,
          feedingDemo);

    ada->switchControllersHack(std::string(""),
      std::string("towards_mouth_bite_transfer_controller"),std::string("inside_mouth_bite_transfer_controller"));

    bool success = moveInFrontOfPerson(
        ada->getArm()->getWorldCollisionConstraint(std::vector<std::string>{"plate", "table", "wheelchair"}),
        personPose,
        distanceToPerson,
        horizontalToleranceForPerson * 2,
        verticalToleranceForPerson * 2,
        feedingDemo);
    ROS_INFO_STREAM("Backward " << success << std::endl);
  }

  std::cout<<"Press [ENTER] to switch controllers:"<<std::endl;
  std::cin.get();std::cout<<"Press [ENTER] again: ";std::cin.get();

  ada->switchControllersHack(std::string("VELOCITY"),
  std::string("move_until_touch_topic_controller"),std::string("towards_mouth_bite_transfer_controller"));

  std::cout<<"Successfully switched controllers!"<<std::endl;

  ada->switchTrajectoryLimits(std::string("acquisition"));

  // // ===== BACK TO PLATE =====
  // ROS_INFO_STREAM("Move back to plate");

  // // TODO: add a back-out motion and then do move above plate with
  // // collisionFree.
  // talk("And now back to the plate.", true);
  // moveAbovePlate(
  //     plate,
  //     plateEndEffectorTransform,
  //     feedingDemo);

  publishTimingDoneToWeb((ros::NodeHandle*)nodeHandle);
}

} // namespace action
} // namespace feeding
