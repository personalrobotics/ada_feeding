#include "feeding/nodes.hpp"
/**
 * Test and debugging BT nodes
 **/

#include <behaviortree_cpp/behavior_tree.h>
#include <iostream>
#include <ros/ros.h>

// Basic Message Types
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>

/// Put Custom Message Types Here:

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

// Subscribe to ROS topic
/// Generic Node Code
template <typename T> class RosSubTopic : public BT::StatefulActionNode {
public:
  RosSubTopic(const std::string &name, const BT::NodeConfig &config,
              ros::NodeHandle *nh)
      : BT::StatefulActionNode(name, config), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::string>("topic"), BT::InputPort<T>("output")};
  }

  BT::NodeStatus onStart() override {
    // Get topic key
    auto topic = getInput<std::string>("topic");
    if (!topic) {
      ROS_WARN_STREAM("Topic Name Not Provided");
      return BT::NodeStatus::FAILURE;
    }
    returnDataReady = false;

    // Create subscriber
    mSub = getSub(topic.value());

    // Check for return value
    return onRunning();
  }

  BT::NodeStatus onRunning() override {
    if (!returnDataReady)
      return BT::NodeStatus::RUNNING;

    setOutput("output", returnData);

    mSub.shutdown();
    return BT::NodeStatus::SUCCESS;
  }

  void onHalted() override {
    // Cancel any hand command
    mSub.shutdown();
  }

private:
  ros::Subscriber getSub(std::string);
  ros::NodeHandle *mNode;
  ros::Subscriber mSub;
  bool returnDataReady;
  T returnData;
};
/// Specializations
//// Set returnDataReady and returnData
template <> ros::Subscriber RosSubTopic<bool>::getSub(std::string topic) {
  return mNode->subscribe<std_msgs::Bool>(
      topic, 1, [&](const std_msgs::BoolConstPtr &msg) {
        returnData = msg->data;
        returnDataReady = true;
      });
}

template <>
ros::Subscriber RosSubTopic<std::string>::getSub(std::string topic) {
  return mNode->subscribe<std_msgs::String>(
      topic, 1, [&](const std_msgs::StringConstPtr &msg) {
        returnData = msg->data;
        returnDataReady = true;
      });
}

template <> ros::Subscriber RosSubTopic<double>::getSub(std::string topic) {
  return mNode->subscribe<std_msgs::Float64>(
      topic, 1, [&](const std_msgs::Float64ConstPtr &msg) {
        returnData = msg->data;
        returnDataReady = true;
      });
}

template <> ros::Subscriber RosSubTopic<int>::getSub(std::string topic) {
  return mNode->subscribe<std_msgs::Int32>(
      topic, 1, [&](const std_msgs::Int32ConstPtr &msg) {
        returnData = msg->data;
        returnDataReady = true;
      });
}

template <>
ros::Subscriber RosSubTopic<std::vector<double>>::getSub(std::string topic) {
  return mNode->subscribe<std_msgs::Float64MultiArray>(
      topic, 1, [&](const std_msgs::Float64MultiArrayConstPtr &msg) {
        returnData = msg->data;
        returnDataReady = true;
      });
}

template <>
ros::Subscriber RosSubTopic<std::vector<int>>::getSub(std::string topic) {
  return mNode->subscribe<std_msgs::Int32MultiArray>(
      topic, 1, [&](const std_msgs::Int32MultiArrayConstPtr &msg) {
        returnData = msg->data;
        returnDataReady = true;
      });
}

/// Put Custom Specializations Below:

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

  // Basic Subscribers
  factory.registerNodeType<RosSubTopic<std::string>>("RosSubString", &nh);
  factory.registerNodeType<RosSubTopic<bool>>("RosSubBool", &nh);
  factory.registerNodeType<RosSubTopic<double>>("RosSubD", &nh);
  factory.registerNodeType<RosSubTopic<int>>("RosSubI", &nh);
  factory.registerNodeType<RosSubTopic<std::vector<int>>>("RosSubVecI", &nh);
  factory.registerNodeType<RosSubTopic<std::vector<double>>>("RosSubVecD", &nh);

  /// Put Custom Subscribers Below:
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
