#include "feeding/nodes.hpp"
/**
 * Nodes for controlling ADA
 **/

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <functional>
#include <vector>

namespace feeding {
namespace nodes {

// Set Velocity or Acceleration Limits
BT::NodeStatus adaSetLimits(BT::TreeNode &self, ada::Ada &robot) {
  // Read Input Ports
  BT::Expected<std::vector<double>> vel =
      self.getInput<std::vector<double>>("velocity");
  Eigen::VectorXd eVel = vel ? Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
                                   vel.value().data(), vel.value().size())
                             : Eigen::VectorXd();
  BT::Expected<std::vector<double>> acc =
      self.getInput<std::vector<double>>("acceleration");
  Eigen::VectorXd eAcc = acc ? Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
                                   acc.value().data(), acc.value().size())
                             : Eigen::VectorXd();

  // Set limits via postprocessor (default Kunz)
  robot.setDefaultPostProcessor(eVel, eAcc);
  return BT::NodeStatus::SUCCESS;
}

// Get Velocity or Acceleration Limits
BT::NodeStatus adaGetLimits(BT::TreeNode &self, ada::Ada &robot) {
  // Read Input Port
  BT::Expected<bool> armInput = self.getInput<bool>("armOnly");
  bool armOnly = armInput ? armInput.value() : false;

  // Get limits from robot
  Eigen::VectorXd eVel = robot.getVelocityLimits(armOnly);
  std::vector<double> vel(eVel.data(), eVel.data() + eVel.size());
  Eigen::VectorXd eAcc = robot.getAccelerationLimits(armOnly);
  std::vector<double> acc(eAcc.data(), eAcc.data() + eAcc.size());

  // Output to Blackboard
  self.setOutput<std::vector<double>>("velocity", vel);
  self.setOutput<std::vector<double>>("acceleration", acc);

  return BT::NodeStatus::SUCCESS;
}

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory,
                          ros::NodeHandle & /*&nh */, ada::Ada &robot) {

  // Velocity and Acceleration Limits
  factory.registerSimpleAction(
      "AdaSetLimits",
      std::bind(adaSetLimits, std::placeholders::_1, std::ref(robot)),
      {BT::InputPort<std::vector<double>>("velocity"),
       BT::InputPort<std::vector<double>>("acceleration")});

  factory.registerSimpleAction(
      "AdaGetLimits",
      std::bind(adaGetLimits, std::placeholders::_1, std::ref(robot)),
      {BT::InputPort<std::vector<double>>("armOnly"),
       BT::OutputPort<std::vector<double>>("velocity"),
       BT::OutputPort<std::vector<double>>("acceleration")});
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
