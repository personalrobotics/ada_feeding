#include "feeding/action/FeedFoodToPerson.hpp"

#include <libada/util.hpp>

#include "feeding/action/Grab.hpp"
#include "feeding/action/MoveAbovePlate.hpp"
#include "feeding/action/MoveDirectlyToPerson.hpp"
#include "feeding/action/MoveInFrontOfPerson.hpp"
#include "feeding/action/MoveTowardsPerson.hpp"
#include "feeding/util.hpp"

using ada::util::createBwMatrixForTSR;
using ada::util::getRosParam;
using aikido::constraint::dart::TSR;

namespace feeding {
namespace action {

static const std::vector<std::string> optionPrompts{"(1) tilt", "(2) no tilt"};
//==============================================================================
void feedFoodToPerson(const std::shared_ptr<Perception> &perception,
                      const Eigen::Isometry3d &plate,
                      const Eigen::Isometry3d &plateEndEffectorTransform,
                      const Eigen::Vector3d *tiltOffset,
                      FeedingDemo *feedingDemo) {
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada> &ada = feedingDemo->getAda();
  const std::shared_ptr<Workspace> &workspace = feedingDemo->getWorkspace();
  const ros::NodeHandle *nodeHandle = feedingDemo->getNodeHandle().get();
  const aikido::constraint::dart::CollisionFreePtr &collisionFree =
      feedingDemo->getCollisionConstraint();
  // const aikido::constraint::dart::CollisionFreePtr&
  // collisionFreeWithWallFurtherBack =
  // feedingDemo->getCollisionConstraintWithWallFurtherBack();
  const Eigen::Isometry3d &personPose = workspace->getPersonPose();
  std::chrono::milliseconds waitAtPerson = feedingDemo->mWaitTimeForPerson;
  bool useSound = feedingDemo->mUseSound;
  double distanceToPerson = feedingDemo->mPersonTSRParameters.at("distance");
  double horizontalToleranceForPerson =
      feedingDemo->mPersonTSRParameters.at("horizontalTolerance");
  double verticalToleranceForPerson =
      feedingDemo->mPersonTSRParameters.at("verticalTolerance");
  double planningTimeout = feedingDemo->mPlanningTimeout;
  int maxNumTrials = feedingDemo->mMaxNumTrials;
  int batchSize = feedingDemo->mBatchSize;
  int maxNumBatches = feedingDemo->mMaxNumBatches;
  int numMaxIterations = feedingDemo->mNumMaxIterations;
  const Eigen::Vector6d &velocityLimits = feedingDemo->mVelocityLimits;

  int feed_food_i = getUserInputWithOptions(std::vector<std::string>{"(1) yourself","(2) neighbor (right)"}, "Who would you like to feed?");

  if  (feed_food_i == 1) {
    auto moveIFOPerson = [&] {
    auto retval = moveInFrontOfPerson(
          ada->getArm()->getWorldCollisionConstraint(), personPose,
          distanceToPerson, horizontalToleranceForPerson,
          verticalToleranceForPerson, feedingDemo);
      return retval;
    };

    bool moveIFOSuccess = false;
    bool moveSuccess = false;
    for (std::size_t i = 0; i < 2; ++i) {
      moveIFOSuccess = moveIFOPerson();
      if (!moveIFOSuccess) {
        ROS_WARN_STREAM("Failed to move in front of person, retry");
        if (useSound) 
          talk("Sorry, I'm having a little trouble moving. Let me try again.");
        continue;
      } else
        break;
    }

    // Send message to web interface to indicate skewer finished
    publishActionDoneToWeb((ros::NodeHandle *)nodeHandle);

    // Ask for Tilt Override
    auto overrideTiltOffset = tiltOffset;

    if (!getRosParam<bool>("/humanStudy/autoTransfer", *nodeHandle)) {
      if (useSound) 
        talk("Should I tilt the food item?", false);
      std::string done = "";
      std::string actionTopic;
      nodeHandle->param<std::string>("/humanStudy/actionTopic", actionTopic,
                                     "/study_action_msgs");
      done = getInputFromTopic(actionTopic, *nodeHandle, false, -1);

      if (done == "tilt_the_food" || done == "tilt") {
        std::vector<double> tiltOffsetVector =
            getRosParam<std::vector<double>>("/study/tiltOffset", *nodeHandle);
        auto tiltOffsetEigen = Eigen::Vector3d(
            tiltOffsetVector[0], tiltOffsetVector[1], tiltOffsetVector[2]);
        overrideTiltOffset = &tiltOffsetEigen;
      } else if (done == "continue") {
        overrideTiltOffset = nullptr;
      } else {
        if (getUserInputWithOptions(optionPrompts,
                                    "Not valid, should I tilt??") == 1) {
          std::vector<double> tiltOffsetVector =
              getRosParam<std::vector<double>>("/study/tiltOffset", *nodeHandle);
          auto tiltOffsetEigen = Eigen::Vector3d(
              tiltOffsetVector[0], tiltOffsetVector[1], tiltOffsetVector[2]);
          overrideTiltOffset = &tiltOffsetEigen;
        } else {
          overrideTiltOffset = nullptr;
        }
      }
    }

    publishTransferDoneToWeb((ros::NodeHandle *)nodeHandle);

    // Check autoTiming, and if false, wait for topic
    if (!getRosParam<bool>("/humanStudy/autoTiming", *nodeHandle)) {
      if (useSound) 
        talk("Let me know when you are ready.", false);
      std::string done = "";
      while (done != "continue") {
        std::string actionTopic;
        nodeHandle->param<std::string>("/humanStudy/actionTopic", actionTopic,
                                       "/study_action_msgs");
        done = getInputFromTopic(actionTopic, *nodeHandle, false, -1);
      }
    } else {
      nodeHandle->setParam("/feeding/facePerceptionOn", true);
      if (useSound) 
        talk("Open your mouth when ready.", false);
      // TODO: Add mouth-open detection.
      // while (true) {
      //   std::this_thread::sleep_for(std::chrono::milliseconds(100));
      //   if (perception->isMouthOpen()) {
      //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
      //     if (perception->isMouthOpen()) {
      //       break;
      //     }
      //   }
      // }
      ada::util::waitForUser("Ready for bite?", ada);

      nodeHandle->setParam("/feeding/facePerceptionOn", false);

      if (getRosParam<bool>("/humanStudy/createError", *nodeHandle)) {
        // Wait an extra 5 seconds
        ROS_WARN_STREAM("Error Requested for Timing!");
        if (useSound) 
          talk("Calculating...");
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
      }
    }

    if (moveIFOSuccess) {

      if (getRosParam<bool>("/humanStudy/autoTransfer", *nodeHandle) &&
          getRosParam<bool>("/humanStudy/createError", *nodeHandle)) {
        ROS_WARN_STREAM("Error Requested for Transfer!");
        // Erroneous Transfer
        /*moveDirectlyToPerson(
          personPose,
          nullptr,
          feedingDemo
          );
        */
        auto trajectory = ada->getArm()->planToConfiguration(
            ada->getArm()->getNamedConfiguration("move_error_pose"),
            ada->getArm()->getSelfCollisionConstraint());
        bool success = true;
        auto future = ada->getArm()->executeTrajectory(
            trajectory); // check velocity limits are set in FeedingDemo + check
                         // success
        try {
          future.get();
        } catch (const std::exception &e) {
          dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
          success = false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        if (useSound) 
          talk("Oops, let me try that again.", true);
        moveIFOSuccess = moveInFrontOfPerson(
            ada->getArm()->getSelfCollisionConstraint(), personPose,
            distanceToPerson, horizontalToleranceForPerson,
            verticalToleranceForPerson, feedingDemo);
      }

      nodeHandle->setParam("/feeding/facePerceptionOn", true);

      if (overrideTiltOffset == nullptr) {
        distanceToPerson = 0;
      }

      ROS_INFO_STREAM("Move towards person");
      moveSuccess =
          moveTowardsPerson(nullptr, perception, distanceToPerson, feedingDemo);
      nodeHandle->setParam("/feeding/facePerceptionOn", false);
    }

    // Execute Tilt
    if (overrideTiltOffset != nullptr) {

      Eigen::Isometry3d person =
          ada->getHand()->getEndEffectorBodyNode()->getTransform();
      person.translation() += *overrideTiltOffset;

      TSR personTSR;
      personTSR.mT0_w = person;

      personTSR.mBw = createBwMatrixForTSR(
          horizontalToleranceForPerson, horizontalToleranceForPerson,
          verticalToleranceForPerson, 0, M_PI / 8, M_PI / 8);
      Eigen::Isometry3d eeTransform = Eigen::Isometry3d::Identity();
      eeTransform.linear() =
          eeTransform.linear() *
          Eigen::Matrix3d(
              Eigen::AngleAxisd(M_PI * -0.25, Eigen::Vector3d::UnitY()) *
              Eigen::AngleAxisd(M_PI * 0.25, Eigen::Vector3d::UnitX()));
      personTSR.mTw_e.matrix() *= eeTransform.matrix();

      // Actually execute movement
      // if (feedingDemo && feedingDemo->getViewer())
      // {
      //   feedingDemo->getViewer()->addTSRMarker(personTSR);
      //   std::cout << "Check TSR" << std::endl;
      //   int n;
      //   std::cin >> n;
      // }
      Eigen::Vector6d slowerVelocity = Eigen::Vector6d(velocityLimits);
      double slowFactor = (velocityLimits[0] > 0.5) ? 2.0 : 1.0;
      slowerVelocity /= slowFactor;

      if (useSound) 
        talk("Tilting, hold tight.", true);

      auto trajectory = ada->getArm()->planToConfiguration(
          ada->getArm()->getNamedConfiguration("home_config"),
          ada->getArm()->getSelfCollisionConstraint());
      bool success = true;
      auto future = ada->getArm()->executeTrajectory(
          trajectory); // check velocity limits are set in FeedingDemo
      try {
        future.get();
      } catch (const std::exception &e) {
        dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
        success = false;
      }

      auto personTSRPtr =
          std::make_shared<aikido::constraint::dart::TSR>(personTSR);
      auto tsr_trajectory = ada->getArm()->planToTSR(
          ada->getEndEffectorBodyNode()->getName(),
          personTSRPtr,
          aikido::robot::util::PlanToTSRParameters(
            maxNumTrials,
            batchSize,
            maxNumBatches,
            numMaxIterations));
      bool tsr_success = true;
      auto tsr_future = ada->getArm()->executeTrajectory(
          tsr_trajectory); // check velocity limits are set in FeedingDemo
      try {
        tsr_future.get();
      } catch (const std::exception &e) {
        dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
        tsr_success = false;
      }

      /*
      Eigen::VectorXd moveTiltPose(6);
      moveTiltPose << -2.9180319979864375, 2.7142495745346644, 2.1617317989753038,
      -3.0472035666546597, -2.144422317154225, -1.1420007596812383;
      ada->moveArmToConfiguration(moveTiltPose, nullptr, 2.0, velocityLimits);
      */
    }

    if (moveIFOSuccess) {
      // ===== EATING =====
      ROS_WARN("Human is eating");
      if (useSound) 
        talk("Ready to eat!");
      std::this_thread::sleep_for(waitAtPerson);

      // Backward
      ada::util::waitForUser("Move backward", ada);
      if (useSound) 
        talk("Let me get out of your way.", true);
      Eigen::Vector3d goalDirection(0, -1, 0);
      bool success = moveInFrontOfPerson(
          ada->getArm()->getWorldCollisionConstraint(
              std::vector<std::string>{"plate", "table", "wheelchair"}),
          personPose, distanceToPerson, horizontalToleranceForPerson * 2,
          verticalToleranceForPerson * 2, feedingDemo);
      ROS_INFO_STREAM("Backward " << success << std::endl);
    }
  } else {
    bool moveIFOSuccess = true;
    auto trajectory = ada->getArm()->planToConfiguration(
        ada->getArm()->getNamedConfiguration("feed_other_person"),
        collisionFree);
    try {
      auto future = ada->getArm()->executeTrajectory(
          trajectory); // check velocity limits are set in FeedingDemo
      future.get();
    } catch (const std::exception &e) {
      dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
      moveIFOSuccess = false;
    }
    // Backward
    ada::util::waitForUser("Move backward", ada);

  }

  
  

  // ===== BACK TO PLATE =====
  ROS_INFO_STREAM("Move back to plate");

  int ifo_pose_i = getUserInputWithOptions(std::vector<std::string>{"(1) above plate","(2) invisible"}, "Pose to move back to?");
    if  (ifo_pose_i == 1) {
      // TODO: add a back-out motion and then do move above plate with
      // collisionFree.
      // if (useSound) 
      //   talk("And now back to the plate.", true);
      moveAbovePlate(plate, plateEndEffectorTransform, feedingDemo);
    } else {
      // NOTE: This is dangerous since we aren't using a constraint! It is done
      // to account for the strange orientation the head is sometimes placed in,
      // after perceiveFace,which messes up collision checking. 
      auto trajectory = ada->getArm()->planToConfiguration(
        ada->getArm()->getNamedConfiguration("invisible_pose"));/*,
        ada->getArm()->getWorldCollisionConstraint());*/
      bool success = true;
      try {
        auto future = ada->getArm()->executeTrajectory(
            trajectory); // check velocity limits are set in FeedingDemo
        future.get();
      } catch (const std::exception &e) {
        dtwarn << "Exception in trajectoryExecution: " << e.what() << std::endl;
        success = false;
      }
    }

  publishTimingDoneToWeb((ros::NodeHandle *)nodeHandle);
}

} // namespace action
} // namespace feeding
