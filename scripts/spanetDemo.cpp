
#include <ros/ros.h>
#include <libada/util.hpp>

#include "feeding/FeedingDemo.hpp"
#include "feeding/util.hpp"
#include "feeding/action/Skewer.hpp"

using ada::util::waitForUser;

namespace feeding {

void spanetDemo(
    FeedingDemo& feedingDemo,
    std::shared_ptr<Perception>& perception,
    ros::NodeHandle nodeHandle)
{

  ROS_INFO_STREAM("========== SPANET DEMO ==========");

  auto ada = feedingDemo.getAda();
  auto workspace = feedingDemo.getWorkspace();
  auto plate = workspace->getPlate()->getRootBodyNode()->getWorldTransform();

  while (true)
  {
    waitForUser("next step?", ada);

    nodeHandle.setParam("/deep_pose/forceFood", false);
    nodeHandle.setParam("/deep_pose/publish_spanet", (true));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ROS_INFO_STREAM("Running spanet demo");

    action::skewer(
    perception,
    &nodeHandle,
    "",
    plate,
    feedingDemo.getPlateEndEffectorTransform(),
    &feedingDemo);

    workspace.reset();
  }

  // ===== DONE =====
  ROS_INFO("Demo finished.");
}
};
