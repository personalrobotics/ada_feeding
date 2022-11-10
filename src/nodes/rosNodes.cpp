#include "feeding/nodes.hpp"
/**
 * Test and debugging BT nodes
 **/

#include <behaviortree_cpp/behavior_tree.h>
#include <iostream>
#include <ros/ros.h>

namespace feeding {
namespace nodes {

// Read ROS Param to Blackboard
// Return FAILURE if param does not exist
// And default port is not used
template <typename T> class RosGetParam : public BT::SyncActionNode {
public:
  RosGetParam(const std::string &name, const BT::NodeConfig &config,
              ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::string>("param"), BT::InputPort<T>("default"),
            BT::OutputPort<T>("target")};
  }

  BT::NodeStatus tick() {
    // Get param key
    auto key = getInput<std::string>("param");
    if (!key) {
      ROS_WARN_STREAM("Param Key Not Provided");
      return BT::NodeStatus::FAILURE;
    }

    // Get Param
    T param;
    if (mNode->getParam(key.value(), param)) {
      // Write to Blackboard
      setOutput<T>("target", param);
      return BT::NodeStatus::SUCCESS;
    } else {
      // Write default to Blackboard
      auto defaultParam = getInput<T>("default");
      if (defaultParam) {
        setOutput<T>("target", defaultParam.value());
        return BT::NodeStatus::SUCCESS;
      }
    }
    // No param or default
    return BT::NodeStatus::FAILURE;
  }

private:
  ros::NodeHandle *mNode;
};

// Write Ros Param from Blackboard
template <typename T> class RosSetParam : public BT::SyncActionNode {
public:
  RosSetParam(const std::string &name, const BT::NodeConfig &config,
              ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::string>("param"), BT::InputPort<T>("value")};
  }

  BT::NodeStatus tick() {
    // Get param key
    auto key = getInput<std::string>("param");
    if (!key) {
      ROS_WARN_STREAM("Param Key Not Provided");
      return BT::NodeStatus::FAILURE;
    }

    // Get new param value
    auto value = getInput<T>("value");
    if (!value) {
      ROS_WARN_STREAM("Param Value Not Provided");
      return BT::NodeStatus::FAILURE;
    }

    // Set Param
    mNode->setParam(key.value(), value.value());
    return BT::NodeStatus::SUCCESS;
  }

private:
  ros::NodeHandle *mNode;
};

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada & /*robot*/) {
  // Get Params
  factory.registerNodeType<RosGetParam<std::string>>("RosGetString", &nh);
  factory.registerNodeType<RosGetParam<bool>>("RosGetBool", &nh);
  factory.registerNodeType<RosGetParam<double>>("RosGetD", &nh);
  factory.registerNodeType<RosGetParam<int>>("RosGetI", &nh);
  factory.registerNodeType<RosGetParam<std::vector<int>>>("RosGetVecI", &nh);
  factory.registerNodeType<RosGetParam<std::vector<double>>>("RosGetVecD", &nh);

  // Set Params
  factory.registerNodeType<RosSetParam<std::string>>("RosSetString", &nh);
  factory.registerNodeType<RosSetParam<bool>>("RosSetBool", &nh);
  factory.registerNodeType<RosSetParam<double>>("RosSetD", &nh);
  factory.registerNodeType<RosSetParam<int>>("RosSetI", &nh);
  factory.registerNodeType<RosSetParam<std::vector<int>>>("RosSetVecI", &nh);
  factory.registerNodeType<RosSetParam<std::vector<double>>>("RosSetVecD", &nh);
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
