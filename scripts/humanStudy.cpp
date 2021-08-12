
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

namespace feeding {

void humanStudyDemo(
    FeedingDemo& feedingDemo,
    std::shared_ptr<Perception>& perception,
    std::shared_ptr<ros::NodeHandle> nodeHandle)
{

  ROS_INFO_STREAM("==========  DEMO ==========");

  auto workspace = feedingDemo.getWorkspace();
  auto plate = workspace->getPlate()->getRootBodyNode()->getWorldTransform();

  //talk("Hello, my name is aid uh. It's my pleasure to serve you today!");

  while (true)
  {
    if (feedingDemo.getFTThresholdHelper())
        feedingDemo.getFTThresholdHelper()->setThresholds(STANDARD_FT_THRESHOLD);

    talk("What food would you like?");
    auto foodName = getUserFoodInput(true, *nodeHandle, true, 30);
    if (foodName == std::string("quit")) {
        break;
    }

    nodeHandle->setParam("/deep_pose/forceFood", false);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ROS_INFO_STREAM("Running human study for " << foodName);

    talk(std::string("One ") + foodName + std::string(" coming right up!"), true);
    /*
    if (foodName == std::string("grape")) {
      foodName = std::string("strawberry");
    }
    */

      bool skewer = action::skewer(
        perception,
        nodeHandle.get(),
        foodName,
        plate,
        feedingDemo.getPlateEndEffectorTransform(),
        &feedingDemo);



      if (!skewer)
      {
        ROS_WARN_STREAM("Restart from the beginning");
        continue;
      }

      // ===== IN FRONT OF PERSON =====
      ROS_INFO_STREAM("Move forque in front of person");
      if (feedingDemo.getFTThresholdHelper())
        feedingDemo.getFTThresholdHelper()->setThresholds(STANDARD_FT_THRESHOLD);

      // TODO: Set tilted explcitly for long food items:
      bool tilted = (foodName == "celery" || foodName == "carrot" || foodName == "bell_pepper" || foodName == "apple");

      action::feedFoodToPerson(
        perception,
        nodeHandle.get(),
        plate,
        feedingDemo.getPlateEndEffectorTransform(),
        tilted ? &feedingDemo.mTiltOffset : nullptr,
        &feedingDemo
        );

  }

  // ===== DONE =====
  ROS_INFO("Demo finished.");
  talk("Thank you, I hope I was helpful!");
}
};
