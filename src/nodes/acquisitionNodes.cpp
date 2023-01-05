#include "feeding/AcquisitionAction.hpp"
#include "feeding/nodes.hpp"
/**
 * Nodes for acquistion action selection
 **/

#include <Eigen/Core>
#include <aikido/perception/DetectedObject.hpp>
#include <behaviortree_cpp/behavior_tree.h>
using aikido::perception::DetectedObject;

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

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory,
                          ros::NodeHandle & /*nh*/, ada::Ada & /*robot*/) {
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
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding