#include "feeding/nodes.hpp"
/**
 * Test and debugging BT nodes
 **/

#include <behaviortree_cpp/behavior_tree.h>
#include <iostream>
#include <ros/ros.h>

#include <std_msgs/Bool.h>

namespace feeding {
namespace nodes {

// Watchdog Check Node
class EStopInterface {
public:
  EStopInterface() : mFirstRun(true) {
    mWatchdogFed.store(false);
    mEStop.store(false);
  }

  void init(ros::NodeHandle &nh) {
    mSub = nh.subscribe<std_msgs::Bool>(
        "watchdog", 1, [this](const std_msgs::BoolConstPtr &msg) {
          if (msg->data)
            mWatchdogFed.store(true);
          else
            mEStop.store(true);
        });
  }

  BT::NodeStatus tick() {
    // If in E-Stop, fail forever
    if (mEStop.load())
      return BT::NodeStatus::FAILURE;

    // On first run, wait for watchdog
    if (mFirstRun) {
      if (!mWatchdogFed.load()) {
        return BT::NodeStatus::RUNNING;
      } else {
        mFirstRun = false;
      }
    }

    // If watchdog not fed, drop to e-stop
    if (!mWatchdogFed.load()) {
      mEStop.store(true);
      return BT::NodeStatus::FAILURE;
    } else {
      mWatchdogFed.store(false);
    }
    return BT::NodeStatus::SUCCESS;
  }

private:
  ros::Subscriber mSub;
  bool mFirstRun;
  std::atomic<bool> mWatchdogFed;
  std::atomic<bool> mEStop;
};

/// Node registration
static EStopInterface estopInterface;
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada & /*robot*/) {
  estopInterface.init(nh);
  factory.registerSimpleCondition(
      "CheckWatchdog",
      [&](BT::TreeNode & /* self */) { return estopInterface.tick(); });
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
