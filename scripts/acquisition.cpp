
#include <aikido/rviz/InteractiveMarkerViewer.hpp>
#include <ros/ros.h>
#include <libada/util.hpp>
#include "feeding/FTThresholdHelper.hpp"
#include "feeding/FeedingDemo.hpp"
#include "feeding/util.hpp"

#include "feeding/perception/Perception.hpp"

using ada::util::getRosParam;
using ada::util::waitForUser;

bool TERMINATE_AT_USER_PROMPT = true;

namespace feeding {

void acquisition(
    FeedingDemo& feedingDemo,
    ros::NodeHandle nodeHandle)
{
  nodeHandle.setParam("/deep_pose/publish_spnet", true);

  feedingDemo.waitForUser("Ready to start.");

  for (int trial = 0; trial < 10; trial++)
  {
    ROS_INFO_STREAM("STARTING TRIAL " << trial << std::endl);

    feedingDemo.skewer("");
  }
  // ===== DONE =====
  feedingDemo.waitForUser("Demo finished.");
}

};
