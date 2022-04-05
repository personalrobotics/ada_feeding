
#include <aikido/rviz/InteractiveMarkerViewer.hpp>
#include <ros/ros.h>
#include <libada/util.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/util.hpp"
#include "feeding/action/PickUpFork.hpp"
#include "feeding/action/PutDownFork.hpp"
#include "feeding/action/FeedFoodToPerson.hpp"
#include "feeding/action/Skewer.hpp"
#include "feeding/action/MoveAbove.hpp"
#include "feeding/action/MoveInFrontOfPerson.hpp"
#include "feeding/action/MoveDirectlyToPerson.hpp"
#include <cstdlib>
#include <ctime>

using ada::util::getRosParam;
using ada::util::waitForUser;

namespace feeding {

void demo(
    FeedingDemo& feedingDemo,
    std::shared_ptr<Perception>& perception,
    ros::NodeHandle nodeHandle)
{

  ROS_INFO_STREAM("==========  DEMO ==========");

  auto workspace = feedingDemo.getWorkspace();
  auto plate = workspace->getPlate()->getRootBodyNode()->getWorldTransform();

  bool useSound = feedingDemo.mUseSound;

  if (useSound) 
    talk("Hello, my name is aid uh. I'm your feeding companion!");

  srand(time(NULL));

  while (true)
  {
    if (feedingDemo.getFTThresholdHelper())
        feedingDemo.getFTThresholdHelper()->setThresholds(STANDARD_FT_THRESHOLD);

    if (useSound)
      talk("What food would you like?", false);

    auto foodName = getUserFoodInput(false, nodeHandle, feedingDemo.mUseAlexa);// "cantaloupe";//
    if (foodName == std::string("quit")) {
        break;
    }

    nodeHandle.setParam("/deep_pose/forceFood", false);
    nodeHandle.setParam("/deep_pose/publish_spnet", (true));
    nodeHandle.setParam("/deep_pose/invertSPNetDirection", false);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ROS_INFO_STREAM("Running bite transfer study for " << foodName);

/*
    if (useSound) {
      switch((rand() % 10)) {
          case 0:
          talk("Good choice!");
          break;
          case 1:
          talk(std::string("Great! I love ") + foodName + std::string("'s!"));
          break;
          case 2:
          talk("Sounds delicious. I wish I had taste buds.");
          break;
          case 4:
          talk("Roger Roger.");
          break;
          case 5:
          talk("Nothing beats fresh fruit.");
          break;
          case 6:
          talk("Nothing escapes my fork!");
          break;
          case 7:
          talk("Thanks Alexa!");
          break;
          default:
          talk("Alright.");
      }
    }
*/

    if (useSound) 
      talk(std::string("One ") + foodName + std::string(" coming right up!"), true);

    // ===== FORQUE PICKUP =====
    if (foodName == "pickupfork")
    {
      action::pickUpFork(
        plate,
        feedingDemo.getPlateEndEffectorTransform(),
        &feedingDemo);
    }
    else if (foodName == "putdownfork")
    {
      action::putDownFork(
        plate,
        feedingDemo.getPlateEndEffectorTransform(),
        &feedingDemo);
    }
    else
    {
      bool skewer = action::skewer(
        perception,
        foodName,
        plate,
        feedingDemo.getPlateEndEffectorTransform(),
        &feedingDemo);

      if (feedingDemo.getFTThresholdHelper())
        feedingDemo.getFTThresholdHelper()->setThresholds(STANDARD_FT_THRESHOLD);

      if (!skewer)
      {
        ROS_WARN_STREAM("Restart from the beginning");
        continue;
      }

      // ===== IN FRONT OF PERSON =====
      ROS_INFO_STREAM("Move forque in front of person");

      auto tiltFoods = feedingDemo.mTiltFoodNames;
      bool tilted = (std::find(tiltFoods.begin(), tiltFoods.end(), foodName) != tiltFoods.end());

      action::feedFoodToPerson(
        perception,
        plate,
        feedingDemo.getPlateEndEffectorTransform(),
        tilted ? &feedingDemo.mTiltOffset : nullptr,
        &feedingDemo
        );
    }
  }

  // ===== DONE =====
  ROS_INFO("Demo finished.");
  if (useSound)
    talk("Thank you, I look forward to feeding you again!");
}
};
