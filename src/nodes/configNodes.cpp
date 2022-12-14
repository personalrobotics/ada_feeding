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

// Write PlanToPose Params from action
class ConfigMoveAbove : public BT::SyncActionNode {
public:
  ConfigMoveAbove(const std::string &name, const BT::NodeConfig &config,
                  ada::Ada *robot, ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mAda(robot), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<DetectedObject>("object"),
            BT::InputPort<Eigen::Isometry3d>("ee_transform"),
            BT::InputPort<bool>("yaw_flip"),
            BT::InputPort<bool>("yaw_agnostic"),
            BT::OutputPort<std::vector<double>>("orig_pos"),
            BT::OutputPort<std::vector<double>>("orig_quat"),
            BT::OutputPort<std::vector<double>>("pos"),
            BT::OutputPort<std::vector<double>>("quat"),
            BT::OutputPort<std::vector<double>>("bounds")};
  }

  BT::NodeStatus tick() override {
    // Read Params
    // Default vertical skewer
    auto transformInput = getInput<Eigen::Isometry3d>("ee_transform");
    Eigen::Isometry3d eeTransform =
        (transformInput) ? transformInput.value()
                         : Eigen::Translation3d(Eigen::Vector3d::UnitZ()) *
                               Eigen::Isometry3d::Identity();

    auto objectInput = getInput<DetectedObject>("objects");
    if (!objectInput) {
      return BT::NodeStatus::FAILURE;
    }
    DetectedObject obj = objectInput.value();

    // Default no yaw flip
    auto yawFlipInput = getInput<bool>("yaw_flip");
    bool yawFlip = yawFlipInput ? yawFlipInput.value() : false;

    // Default normal yaw bounds
    auto yawAgnosticInput = getInput<bool>("yaw_agnostic");
    bool yawAgnostic = yawAgnosticInput ? yawAgnosticInput.value() : false;

    // Read Ros Params
    double distance;
    if (!mNode->getParam("move_above/distance", distance)) {
      ROS_WARN_STREAM("ConfigMoveAbove: Need distance param");
      return BT::NodeStatus::FAILURE;
    }

    std::vector<double> bounds{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    mNode->getParam("move_above/tsr_bounds", bounds);
    if (bounds.size() != 6) {
      ROS_WARN_STREAM("ConfigMoveAbove: TSR bounds must be size 6");
      return BT::NodeStatus::FAILURE;
    }
    // If yaw agnostic, unbounded
    if (yawAgnostic)
      bounds[5] = M_PI;
    setOutput<std::vector<double>>("bounds", bounds);

    // Origin is the food item
    Eigen::Isometry3d objTransform =
        obj.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();

    // "Flatten" orientation, so Z is facing straight up
    // We do this by taking the X axis of the object frame
    // and projecting it onto the X/Y plane of the world
    // frame. This is the X-axis of the new origin frame
    Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();
    origin.translation() = objTransform.translation();
    Eigen::Vector3d flatX = objTransform.rotation() * Eigen::Vector3d::UnitX();
    Eigen::AngleAxisd zRotation =
        Eigen::AngleAxisd(atan2(flatX[1], flatX[0]), Eigen::Vector3d::UnitZ());
    origin.linear() = origin.linear() * zRotation;
    // If yaw flip, rotate pi about Z
    if (yawFlip)
      origin.linear() =
          origin.linear() * Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ());

    // Output origin frame
    Eigen::Vector3d eOrigPos = origin.translation();
    std::vector<double> orig_pos{eOrigPos[0], eOrigPos[1], eOrigPos[2]};
    Eigen::Quaterniond eOrigQuat(origin.linear());
    std::vector<double> orig_quat{eOrigQuat.w(), eOrigQuat.x(), eOrigQuat.y(),
                                  eOrigQuat.z()};
    setOutput<std::vector<double>>("orig_pos", orig_pos);
    setOutput<std::vector<double>>("orig_quat", orig_quat);

    // Desired pose relative to origin
    // Scale by distance param
    eeTransform.translation() =
        distance * eeTransform.translation().normalized();

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

class ConfigMoveInto : public BT::SyncActionNode {
public:
  ConfigMoveInto(const std::string &name, const BT::NodeConfig &config,
                 ada::Ada *robot, ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mAda(robot), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<DetectedObject>("object"),
            BT::InputPort<double>("overshoot"),
            BT::InputPort<std::vector<double>>("obj_off"),
            BT::OutputPort<std::vector<double>>("offset")};
  }

  BT::NodeStatus tick() override {
    // Read Params
    auto objectInput = getInput<DetectedObject>("object");
    if (!objectInput) {
      return BT::NodeStatus::FAILURE;
    }
    // Just select the first object
    // TODO: more intelligent object selection
    DetectedObject obj = objectInput.value();

    // Input other arguments
    auto overshootInput = getInput<double>("overshoot");
    double overshoot = overshootInput ? overshootInput.value() : 0.0;
    auto offInput = getInput<std::vector<double>>("obj_off");
    std::vector<double> off =
        offInput ? offInput.value() : std::vector<double>{0.0, 0.0, 0.0};
    Eigen::Vector3d eOff = Eigen::Vector3d::Zero();
    eOff << off[0], off[1], off[2];

    // Compute offset
    Eigen::Isometry3d objTransform =
        obj.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
    Eigen::Isometry3d eeTransform =
        mAda->getEndEffectorBodyNode()->getWorldTransform();
    Eigen::Vector3d eOffset =
        objTransform.translation() - eeTransform.translation();
    // Add overshoot and additional offset
    eOffset = eOffset.normalized() * (eOffset.norm() + overshoot) + eOff;

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
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
