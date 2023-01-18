#include "feeding/AcquisitionAction.hpp"
#include "feeding/nodes.hpp"
/**
 * Nodes for acquistion action selection
 **/

#include <Eigen/Core>
#include <aikido/perception/DetectedObject.hpp>
#include <behaviortree_cpp/behavior_tree.h>
using aikido::perception::DetectedObject;

#ifdef POSTHOC_FOUND
// Online Learning Headers
#include <posthoc_learn/GetAction.h>
#include <posthoc_learn/PublishLoss.h>
#endif

namespace feeding {
namespace nodes {

// Unpack AcquistiionAction Object to Blackboard
BT::NodeStatus UnpackAction(BT::TreeNode &self) {
  auto actionInput = self.getInput<AcquisitionAction>("action");
  if (!actionInput)
    return BT::NodeStatus::FAILURE;
  auto action = actionInput.value();

  self.setOutput<Eigen::Isometry3d>("pre_transform", action.pre_transform);
  self.setOutput<Eigen::Vector3d>("pre_offset", action.pre_offset);
  self.setOutput<double>("pre_force", action.pre_force);
  self.setOutput<double>("pre_torque", action.pre_torque);
  self.setOutput<Eigen::Vector3d>("grasp_offset", action.grasp_offset);
  self.setOutput<Eigen::Vector3d>("grasp_rot", action.grasp_rot);
  self.setOutput<double>("grasp_duration", action.grasp_duration);
  self.setOutput<double>("grasp_force", action.grasp_force);
  self.setOutput<double>("grasp_torque", action.grasp_torque);
  self.setOutput<Eigen::Vector3d>("ext_offset", action.ext_offset);
  self.setOutput<Eigen::Vector3d>("ext_rot", action.ext_rot);
  self.setOutput<double>("ext_duration", action.ext_duration);
  self.setOutput<double>("ext_force", action.ext_force);
  self.setOutput<double>("ext_torque", action.ext_torque);
  self.setOutput<double>("rotate_pref", action.rotate_pref);

  return BT::NodeStatus::SUCCESS;
}

// Get Default AcquisitonAction (i.e. vertical skewer)
BT::NodeStatus DefaultAction(BT::TreeNode &self) {
  self.setOutput<AcquisitionAction>("target", AcquisitionAction());
  return BT::NodeStatus::SUCCESS;
}

// Get AcquisitonAction from Library
BT::NodeStatus GetAction(BT::TreeNode &self) {
  auto libInput = self.getInput<XmlRpc::XmlRpcValue>("library");
  if (!libInput)
    return BT::NodeStatus::FAILURE;
  auto library = AcquisitionAction::fromRosParam(libInput.value());
  if (library.size() < 1)
    return BT::NodeStatus::FAILURE;
  auto idxInput = self.getInput<int>("index");
  auto index = (idxInput) ? idxInput.value() : 0;
  if (index < 0 || (size_t)index >= library.size())
    return BT::NodeStatus::FAILURE;
  self.setOutput<AcquisitionAction>("target", library[index]);
  return BT::NodeStatus::SUCCESS;
}

#ifdef POSTHOC_FOUND

/// Get action from online learning server
class GetActionOnline : public BT::SyncActionNode {
public:
  GetActionOnline(const std::string &name, const BT::NodeConfig &config,
                  ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<sensor_msgs::Image>("context"),
            BT::OutputPort<int>("action_index"),
            BT::OutputPort<std::vector<double>>("probabilities")};
  }

  BT::NodeStatus tick() override {
    // Get Input args
    auto imageInput = getInput<sensor_msgs::Image>("context");
    if (!imageInput)
      return BT::NodeStatus::FAILURE;

    // Read Ros Params
    std::string server;
    if (!mNode->getParam("posthoc/get_action", server)) {
      ROS_WARN_STREAM("GetActionOnline: Need action service");
      return BT::NodeStatus::FAILURE;
    }
    ros::ServiceClient client =
        mNode->serviceClient<posthoc_learn::GetAction>(server);

    // Call service
    posthoc_learn::GetAction srv;
    srv.request.image = imageInput.value();
    if (!client.call(srv)) {
      ROS_ERROR("Failed to call service get_action");
      return BT::NodeStatus::FAILURE;
    }
    setOutput<std::vector<double>>("probabilities", srv.response.p_t);
    setOutput<int>("action_index", srv.response.a_t);
    return BT::NodeStatus::SUCCESS;
  }

private:
  ros::NodeHandle *mNode;
};

/// Report Loss to Online Learning Server
class PublishLossOnline : public BT::SyncActionNode {
public:
  PublishLossOnline(const std::string &name, const BT::NodeConfig &config,
                    ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<sensor_msgs::Image>("context"),
            BT::InputPort<std::vector<double>>("posthoc"),
            BT::InputPort<int>("action_index"),
            BT::InputPort<std::vector<double>>("probabilities"),
            BT::InputPort<double>("loss")};
  }

  BT::NodeStatus tick() override {
    // Get Input args
    auto imageInput = getInput<sensor_msgs::Image>("context");
    if (!imageInput)
      return BT::NodeStatus::FAILURE;

    auto posthocInput = getInput<std::vector<double>>("posthoc");
    if (!posthocInput)
      return BT::NodeStatus::FAILURE;

    auto actionInput = getInput<int>("action_index");
    if (!actionInput)
      return BT::NodeStatus::FAILURE;

    auto probInput = getInput<std::vector<double>>("probabilities");
    if (!probInput)
      return BT::NodeStatus::FAILURE;

    auto lossInput = getInput<double>("loss");
    if (!lossInput)
      return BT::NodeStatus::FAILURE;

    // Read Ros Params
    std::string server;
    if (!mNode->getParam("posthoc/publish_loss", server)) {
      ROS_WARN_STREAM("PublishLossOnline: Need loss service");
      return BT::NodeStatus::FAILURE;
    }
    ros::ServiceClient client =
        mNode->serviceClient<posthoc_learn::PublishLoss>(server);

    // Call service
    posthoc_learn::PublishLoss srv;
    srv.request.image = imageInput.value();
    srv.request.haptic = posthocInput.value();
    srv.request.p_t = probInput.value();
    srv.request.a_t = actionInput.value();
    srv.request.loss = lossInput.value();
    if (!client.call(srv) || !srv.response.success) {
      ROS_ERROR("Failed to call service publish_loss");
      return BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::SUCCESS;
  }

private:
  ros::NodeHandle *mNode;
};
#endif

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada & /*robot*/) {
  factory.registerSimpleAction(
      "AcquisitionUnpackAction", std::bind(UnpackAction, std::placeholders::_1),
      {BT::InputPort<AcquisitionAction>("action"),
       BT::OutputPort<double>("rotate_pref"),
       BT::OutputPort<Eigen::Isometry3d>("pre_transform"),
       BT::OutputPort<Eigen::Vector3d>("pre_offset"),
       BT::OutputPort<double>("pre_force"),
       BT::OutputPort<double>("pre_torque"),
       BT::OutputPort<Eigen::Vector3d>("grasp_offset"),
       BT::OutputPort<Eigen::Vector3d>("grasp_rot"),
       BT::OutputPort<double>("grasp_duration"),
       BT::OutputPort<double>("grasp_force"),
       BT::OutputPort<double>("grasp_torque"),
       BT::OutputPort<Eigen::Vector3d>("ext_offset"),
       BT::OutputPort<Eigen::Vector3d>("ext_rot"),
       BT::OutputPort<double>("ext_duration"),
       BT::OutputPort<double>("ext_force"),
       BT::OutputPort<double>("ext_torque")});

  factory.registerSimpleAction("AcquisitionDefaultAction",
                               std::bind(DefaultAction, std::placeholders::_1),
                               {BT::OutputPort<AcquisitionAction>("target")});

  factory.registerSimpleAction("AcquisitionGetAction",
                               std::bind(GetAction, std::placeholders::_1),
                               {BT::InputPort<XmlRpc::XmlRpcValue>("library"),
                                BT::InputPort<int>("index"),
                                BT::OutputPort<AcquisitionAction>("target")});

#ifdef POSTHOC_FOUND
  factory.registerNodeType<GetActionOnline>("AcquisitionGetActionOnline", &nh);
  factory.registerNodeType<PublishLossOnline>("AcquisitionPublishLossOnline",
                                              &nh);
#endif
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding