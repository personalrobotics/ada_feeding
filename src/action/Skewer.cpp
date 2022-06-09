#include "feeding/action/Skewer.hpp"

#include <libada/util.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/action/DetectAndMoveAboveFood.hpp"
#include "feeding/action/Grab.hpp"
#include "feeding/action/MoveAbovePlate.hpp"
#include "feeding/action/MoveInto.hpp"
#include "feeding/action/MoveOutOf.hpp"
#include "feeding/util.hpp"

using ada::util::getRosParam;

static const std::vector<std::string> optionPrompts{"(1) success", "(2) fail"};
static const std::vector<std::string> actionPrompts{"(1) skewer", "(3) tilt",
                                                    "(5) angle"};

namespace feeding {
namespace action {

//==============================================================================
bool skewer(const std::shared_ptr<Perception> &perception,
            const std::string &foodName, const Eigen::Isometry3d &plate,
            const Eigen::Isometry3d &plateEndEffectorTransform,
            FeedingDemo *feedingDemo) {
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada> &ada = feedingDemo->getAda();
  const ros::NodeHandle *nodeHandle = feedingDemo->getNodeHandle().get();
  const aikido::constraint::dart::CollisionFreePtr &collisionFree =
      feedingDemo->getCollisionConstraint();
  const std::unordered_map<std::string, double> &foodSkeweringForces =
      feedingDemo->mFoodSkeweringForces;
  bool useSound = feedingDemo->mUseSound;
  double heightAboveFood = feedingDemo->mFoodTSRParameters.at("height");
  double rotationToleranceForFood =
      feedingDemo->mFoodTSRParameters.at("rotationTolerance");
  double moveOutofFoodLength = feedingDemo->mMoveOufOfFoodLength;
  std::chrono::milliseconds waitTimeForFood = feedingDemo->mWaitTimeForFood;
  // const Eigen::Vector6d& velocityLimits = feedingDemo.mVelocityLimits;
  const std::shared_ptr<FTThresholdHelper> &ftThresholdHelper =
      feedingDemo->getFTThresholdHelper();
  std::vector<std::string> rotationFreeFoodNames =
      feedingDemo->mRotationFreeFoodNames;

  // TODO (egordon): add option to disable override in feeding_demo.yaml
  int actionOverride = -1;
  if (feedingDemo->mPickUpAngleModes.count(foodName)) {
    actionOverride = feedingDemo->mPickUpAngleModes[foodName];
  }

  ROS_INFO_STREAM("Move above plate");
  bool abovePlaceSuccess =
      moveAbovePlate(plate, plateEndEffectorTransform, feedingDemo);

  if (!abovePlaceSuccess) {
    if (useSound) 
      talk(
          "Sorry, I'm having a little trouble moving. Mind if I get a little "
          "help?");
    ROS_WARN_STREAM("Move above plate failed. Please restart");
    return false;
  }

  bool detectAndMoveAboveFoodSuccess = true;

  if (!getRosParam<bool>("/humanStudy/autoAcquisition",
                         *(feedingDemo->getNodeHandle()))) {
    // Read Action from Topic
    if (useSound) 
      talk("How should I pick up the food?", true);
    ROS_INFO_STREAM("Waiting for action...");
    std::string actionName;
    std::string actionTopic;
    feedingDemo->getNodeHandle()->param<std::string>(
        "/humanStudy/actionTopic", actionTopic, "/study_action_msgs");
    actionName = getInputFromTopic(actionTopic, *(feedingDemo->getNodeHandle()),
                                   false, -1);
    if (useSound) 
      talk("Alright, let me use " + actionName, false);


    if (actionName == "skewer") {
      actionOverride = 1;
    } else if (actionName == "vertical") {
      actionOverride = 1;
    } else if (actionName == "cross_skewer") {
      actionOverride = 1;
    } else if (actionName == "tilt") {
      actionOverride = 3;
    } else if (actionName == "cross_tilt") {
      actionOverride = 3;
    } else if (actionName == "angle") {
      actionOverride = 5;
    } else if (actionName == "cross_angle") {
      actionOverride = 5;
    } else {
      actionOverride = getUserInputWithOptions(
          actionPrompts, "Didn't get valid action. Choose manually:");
      if (actionOverride > 5 || actionOverride < 0) {
        actionOverride = 1;
      }
    }
  }

  if (std::find(rotationFreeFoodNames.begin(), rotationFreeFoodNames.end(),
                foodName) != rotationFreeFoodNames.end()) {
    rotationToleranceForFood = M_PI;
    if (actionOverride == 1) {
      actionOverride = 0;
    }
  } else {
    // Read Action from FeedingDemo
  }

  // Pause a bit so camera can catch up
  /*
  if(velocityLimits[0] > 0.5) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  */
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  for (std::size_t trialCount = 0; trialCount < 3; ++trialCount) {

    Eigen::Vector3d endEffectorDirection(0, 0, -1);
    std::unique_ptr<FoodItem> item;
    for (std::size_t i = 0; i < 2; ++i) {
      if (i == 0) {
        if (useSound) 
          talk(std::string("Planning to the ") + foodName, true);
      }
      if (i == 1) {
        // if (useSound) 
        //   talk("Adjusting, hold tight!", true);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Set Action Override
        auto action = item->getAction();
        int actionNum = 0;
        switch (action->getTiltStyle()) {
        case TiltStyle::ANGLED:
          actionNum = 4;
          break;
        case TiltStyle::VERTICAL:
          actionNum = 2;
          break;
        default:
          actionNum = 0;
        }
        if (action->getRotationAngle() > 0.01) {
          // Assume 90-degree action
          actionNum++;
        }
        // Call here so we don't overwrite features
        Eigen::Vector3d foodVec =
            item->getPose().rotation() * Eigen::Vector3d::UnitX();
        double baseRotateAngle = atan2(foodVec[1], foodVec[0]);
        // detectAndMoveAboveFood(perception, foodName, rotationToleranceForFood,
        //                        feedingDemo, &baseRotateAngle, actionNum);
        auto tiltStyle = item->getAction()->getTiltStyle();
        if (tiltStyle == TiltStyle::ANGLED) {
          // Apply base rotation of food
          Eigen::Isometry3d eePose =
              ada->getHand()->getEndEffectorBodyNode()->getTransform();
          Eigen::Vector3d newEEDir =
              eePose.rotation() * Eigen::Vector3d::UnitZ();
          newEEDir[2] = sqrt(pow(newEEDir[0], 2.0) + pow(newEEDir[1], 2.0)) *
                        (-0.2 / 0.1);
          endEffectorDirection = newEEDir;
          endEffectorDirection.normalize();

          Eigen::Vector3d forkXAxis =
              eePose.rotation() * Eigen::Vector3d::UnitX();
          forkXAxis[2] = 0.0;
          forkXAxis.normalize();
          endEffectorDirection *= heightAboveFood;
          // endEffectorDirection += (0.01 * forkXAxis);
          endEffectorDirection.normalize();
        } else if (tiltStyle == TiltStyle::NONE) {
          // Apply base rotation of food
          Eigen::Isometry3d eePose =
              ada->getHand()->getEndEffectorBodyNode()->getTransform();
          Eigen::Vector3d forkYAxis =
              eePose.rotation() * Eigen::Vector3d::UnitY();
          forkYAxis[2] = 0.0;
          forkYAxis.normalize();
          Eigen::Vector3d forkXAxis =
              eePose.rotation() * Eigen::Vector3d::UnitX();
          forkXAxis[2] = 0.0;
          forkXAxis.normalize();
          endEffectorDirection *= heightAboveFood;
          endEffectorDirection.normalize();
        } else if (tiltStyle == TiltStyle::VERTICAL) {
          // Apply base rotation of food
          Eigen::Isometry3d eePose =
              ada->getHand()->getEndEffectorBodyNode()->getTransform();
          Eigen::Vector3d forkYAxis =
              eePose.rotation() * Eigen::Vector3d::UnitY();
          Eigen::Vector3d forkXAxis =
              eePose.rotation() * Eigen::Vector3d::UnitX();
          forkXAxis[2] = 0.0;
          forkXAxis.normalize();
          forkYAxis[2] = 0.0;
          forkYAxis.normalize();
          endEffectorDirection *= heightAboveFood;
          // endEffectorDirection += ((-0.025 * forkYAxis) + (-0.01 * forkXAxis));
          endEffectorDirection.normalize();
        }
        break;
      }

      ROS_INFO_STREAM("Detect and Move above food");
      item =
          detectAndMoveAboveFood(perception, foodName, rotationToleranceForFood,
                                 feedingDemo, nullptr, actionOverride);

      if (!item) {
        if (useSound) 
          talk("Failed, let me start from the beginning");
        return false;
      }

      if (!item) {
        detectAndMoveAboveFoodSuccess = false;
      }

      // Add error if autonomous
      if (getRosParam<bool>("/humanStudy/autoAcquisition",
                            *(feedingDemo->getNodeHandle())) && // autonomous
          getRosParam<bool>("/humanStudy/createError",
                            *(feedingDemo->getNodeHandle())) && // add error
          trialCount == 0)                                      // First Trial
      {
        ROS_WARN_STREAM("Error Requested for Acquisition!");
        endEffectorDirection(1) -= 1.0;
        endEffectorDirection.normalize();
      }
    }

    if (!detectAndMoveAboveFoodSuccess)
      return false;

    ROS_INFO_STREAM("Getting " << foodName << "with "
                               << foodSkeweringForces.at(foodName)
                               << "N with angle mode ");

    double torqueThreshold = 2;
    if (ftThresholdHelper)
      ftThresholdHelper->setThresholds(foodSkeweringForces.at(foodName),
                                       torqueThreshold);

    // ===== INTO FOOD =====
    // if (useSound) 
    //   talk("Here we go!", true);

    // Set end effector to move into food
    endEffectorDirection = perception->getTrackedFoodItemPose().translation() - ada->getEndEffectorBodyNode()->getTransform().translation();
    endEffectorDirection.normalize();

    auto moveIntoSuccess = moveInto(perception, TargetItem::FOOD,
                                    endEffectorDirection, feedingDemo);

    if (!moveIntoSuccess) {
      ROS_INFO_STREAM("Failed. Retry");
      if (useSound) 
        talk("Sorry, I'm having a little trouble moving. Let me try again.");
      return false;
    }

    std::this_thread::sleep_for(waitTimeForFood);

    // ===== OUT OF FOOD =====
    Eigen::Vector3d direction(0, 0, 1);
    moveOutOf(nullptr, TargetItem::FOOD, moveOutofFoodLength * 2.0, direction,
              feedingDemo);

    if (getUserInputWithOptions(optionPrompts, "Did I succeed?") ==
        1) // true)//
    {
      ROS_INFO_STREAM("Successful");
      if (useSound) 
        talk("Success.");
      return true;
    }

    ROS_INFO_STREAM("Failed.");
    if (useSound) 
      talk("Failed, let me try again.");
  }
  return false;
}

} // namespace action
} // namespace feeding
