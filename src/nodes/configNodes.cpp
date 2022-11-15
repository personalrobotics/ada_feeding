#include "feeding/nodes.hpp"
/**
 * Nodes for configuring movement actions
 **/

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <cmath>
#include <iostream>

#include <aikido/perception/DetectedObject.hpp>
using aikido::perception::DetectedObject;

namespace feeding {
namespace nodes {

class ConfigMoveAbove : public BT::SyncActionNode {
public:
  ConfigMoveAbove(const std::string &name, const BT::NodeConfig &config,
                  ada::Ada *robot, ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mAda(robot), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::vector<DetectedObject>>("objects"),
            BT::InputPort<std::vector<double>>("action"),
            BT::OutputPort<std::vector<double>>("orig_pos"),
            BT::OutputPort<std::vector<double>>("orig_quat"),
            BT::OutputPort<std::vector<double>>("pos"),
            BT::OutputPort<std::vector<double>>("quat"),
            BT::OutputPort<std::vector<double>>("bounds")};
  }

  BT::NodeStatus tick() override {
    // Read Params
    auto actionInput = getInput<std::vector<double>>("action");
    long action = (actionInput && actionInput.value().size() > 0)
                      ? std::lround(actionInput.value()[0])
                      : 0L;

    auto objectInput = getInput<std::vector<DetectedObject>>("objects");
    if (!objectInput || objectInput.value().size() < 1) {
      return BT::NodeStatus::FAILURE;
    }
    // Just select the first object
    // TODO: more intelligent object selection
    DetectedObject obj = objectInput.value()[0];

    // Read Ros Params
    double height;
    if (!mNode->getParam("move_above/height", height)) {
      ROS_WARN_STREAM("ConfigMoveAbove: Need height param");
      return BT::NodeStatus::FAILURE;
    }
    std::vector<std::string> ra;
    mNode->getParam("move_above/rotation_agnostic", ra);
    bool rotationFree =
        (std::find(std::begin(ra), end(ra), obj.getName()) != std::end(ra));

    std::vector<double> bounds{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mNode->getParam("move_above/tsr_bounds", bounds);
    if (bounds.size() != 6) {
      ROS_WARN_STREAM("ConfigMoveAbove: TSR bounds must be size 6");
      return BT::NodeStatus::FAILURE;
    }
    // If rotation free, yaw is unbounded
    if (rotationFree)
      bounds[5] = M_PI;
    setOutput<std::vector<double>>("bounds", bounds);

    // Origin is the food item
    Eigen::Isometry3d objTransform =
        obj.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();

    // "Flatten" orientation, so Z is facing straight up
    // We do this by taking the X axis of the object frame
    // and projecting it onto the X/Y plane of the world frame.
    // This is the X-axis of the new origin frame
    Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();
    origin.translation() = objTransform.translation();
    Eigen::Vector3d flatX = objTransform.rotation() * Eigen::Vector3d::UnitX();
    Eigen::AngleAxisd zRotation =
        Eigen::AngleAxisd(atan2(flatX[1], flatX[0]), Eigen::Vector3d::UnitZ());
    origin.linear() = origin.linear() * zRotation;

    // Output origin frame
    Eigen::Vector3d eOrigPos = origin.translation();
    std::vector<double> orig_pos{eOrigPos[0], eOrigPos[1], eOrigPos[2]};
    Eigen::Quaterniond eOrigQuat(origin.linear());
    std::vector<double> orig_quat{eOrigQuat.w(), eOrigQuat.x(), eOrigQuat.y(),
                                  eOrigQuat.z()};
    setOutput<std::vector<double>>("orig_pos", orig_pos);
    setOutput<std::vector<double>>("orig_quat", orig_quat);

    // Desired pose relative to origin
    Eigen::Isometry3d eeTransform = Eigen::Isometry3d::Identity();
    eeTransform.translation()[2] = height;
    // Flip so Z (i.e. EE) is facing down
    eeTransform.linear() = eeTransform.linear() *
                           Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY());

    // Apply Z rotation for food alignment
    double rotationAngle = 0;
    switch (action) {
    case 1L:
    case 3L:
    case 5L:
      rotationAngle = M_PI / 2.0;
    }
    eeTransform.linear() =
        eeTransform.linear() *
        Eigen::AngleAxisd(rotationAngle, Eigen::Vector3d::UnitZ());

    // Tilted Vertical Rotation (i.e. vertical tines)
    switch (action) {
    case 2L:
    case 3L:
      eeTransform.linear() = eeTransform.linear() *
                             Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitX());
    }

    // Tilted Angled Rotation + Translation (offset)
    switch (action) {
    case 4L:
    case 5L:
      eeTransform.linear() =
          eeTransform.linear() *
          Eigen::AngleAxisd(-M_PI / 8, Eigen::Vector3d::UnitX());
      eeTransform.translation() =
          Eigen::AngleAxisd(
              -rotationAngle,
              Eigen::Vector3d::UnitZ()) // Take into account action rotation
          * Eigen::Vector3d{0, -sin(M_PI * 0.25) * height * 0.7,
                            cos(M_PI * 0.25) * height * 0.9};
    }

    // Output EE target pose
    Eigen::Vector3d ePos = eeTransform.translation();
    std::vector<double> pos{ePos[0], ePos[1], ePos[2]};
    Eigen::Quaterniond eQuat(eeTransform.linear());
    std::vector<double> quat{eQuat.w(), eQuat.x(), eQuat.y(), eQuat.z()};
    setOutput<std::vector<double>>("pos", pos);
    setOutput<std::vector<double>>("quat", quat);

    return BT::NodeStatus::SUCCESS;
  }

private:
  ada::Ada *mAda;
  ros::NodeHandle *mNode;
};

/// Action Selection
BT::NodeStatus ConfigActionSelect(BT::TreeNode &self, ros::NodeHandle &nh) {
  // Input Param
  auto objectInput = self.getInput<std::vector<DetectedObject>>("foods");
  if (!objectInput || objectInput.value().size() < 1) {
    return BT::NodeStatus::FAILURE;
  }
  // Just select the first object
  // TODO: more intelligent object selection
  DetectedObject obj = objectInput.value()[0];

  // Ros Params
  std::vector<std::string> foodNames;
  nh.getParam("action_selection/food_names", foodNames);

  std::vector<double> actions;
  nh.getParam("action_selection/actions", actions);

  if (foodNames.size() != actions.size()) {
    ROS_WARN_STREAM(
        "ConfigActionSelect: action and foodName params must be same size");
    return BT::NodeStatus::FAILURE;
  }

  // Search for food name
  std::vector<double> ret{1.0};
  auto it = std::find(foodNames.begin(), foodNames.end(), obj.getName());
  if (it != foodNames.end()) {
    int index = it - foodNames.begin();
    ret[0] = actions[index];
  } else {
    ROS_WARN_STREAM("Using default action for: " << obj.getName());
  }

  self.setOutput("action", ret);
  return BT::NodeStatus::SUCCESS;
}

class ConfigMoveInto : public BT::SyncActionNode {
public:
  ConfigMoveInto(const std::string &name, const BT::NodeConfig &config,
                 ada::Ada *robot, ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mAda(robot), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::vector<DetectedObject>>("objects"),
            BT::InputPort<double>("overshoot"),
            BT::OutputPort<std::vector<double>>("offset")};
  }

  BT::NodeStatus tick() override {
    // Read Params
    auto objectInput = getInput<std::vector<DetectedObject>>("objects");
    if (!objectInput || objectInput.value().size() < 1) {
      return BT::NodeStatus::FAILURE;
    }
    // Just select the first object
    // TODO: more intelligent object selection
    DetectedObject obj = objectInput.value()[0];

    auto overshootInput = getInput<double>("overshoot");
    double overshoot = overshootInput ? overshootInput.value() : 0.0;

    Eigen::Isometry3d objTransform =
        obj.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
    Eigen::Isometry3d eeTransform =
        mAda->getEndEffectorBodyNode()->getWorldTransform();
    Eigen::Vector3d eOffset =
        objTransform.translation() - eeTransform.translation();
    // Add overshoot
    eOffset = eOffset.normalized() * (eOffset.norm() + overshoot);

    std::vector<double> offset{eOffset.x(), eOffset.y(), eOffset.z()};

    setOutput("offset", offset);
    return BT::NodeStatus::SUCCESS;
  }

private:
  ada::Ada *mAda;
  ros::NodeHandle *mNode;
};

class ConfigMoveToFace : public BT::SyncActionNode {
public:
  ConfigMoveToFace(const std::string &name, const BT::NodeConfig &config,
                   ada::Ada *robot, ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mAda(robot), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::vector<DetectedObject>>("faces"),
            BT::InputPort<double>("undershoot"),
            BT::OutputPort<std::vector<double>>("offset")};
  }

  BT::NodeStatus tick() override {
    // Read Params
    auto objectInput = getInput<std::vector<DetectedObject>>("faces");
    if (!objectInput || objectInput.value().size() < 1) {
      return BT::NodeStatus::FAILURE;
    }
    // Just select the first object
    // TODO: more intelligent object selection
    DetectedObject obj = objectInput.value()[0];

    auto undershootInput = getInput<double>("undershoot");
    double undershoot = undershootInput ? undershootInput.value() : 0.0;

    Eigen::Isometry3d objTransform =
        obj.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
    Eigen::Isometry3d eeTransform =
        mAda->getEndEffectorBodyNode()->getWorldTransform();
    Eigen::Vector3d eOffset =
        objTransform.translation() - eeTransform.translation();
    // Add overshoot
    eOffset = eOffset.normalized() * (eOffset.norm() - undershoot);

    std::vector<double> offset{eOffset.x(), eOffset.y(), eOffset.z()};

    setOutput("offset", offset);
    return BT::NodeStatus::SUCCESS;
  }

private:
  ada::Ada *mAda;
  ros::NodeHandle *mNode;
};

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada &robot) {
  factory.registerNodeType<ConfigMoveAbove>("ConfigMoveAbove", &robot, &nh);
  factory.registerNodeType<ConfigMoveInto>("ConfigMoveInto", &robot, &nh);
  factory.registerNodeType<ConfigMoveToFace>("ConfigMoveToFace", &robot, &nh);
  factory.registerSimpleAction(
      "ConfigActionSelect",
      std::bind(ConfigActionSelect, std::placeholders::_1, std::ref(nh)),
      {BT::InputPort<std::vector<DetectedObject>>("foods"),
       BT::OutputPort<std::vector<double>>("action")});
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
