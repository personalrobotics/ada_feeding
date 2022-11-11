#include "feeding/nodes.hpp"
/**
 * Nodes for controlling ADA
 **/

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <functional>
#include <iostream>
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

// Ada Hand Manipulation nodes
enum AdaHandNodeType { kOPEN, kCLOSE, kPRESHAPE, kCONFIG };

template <AdaHandNodeType T> class AdaHandNode : public BT::StatefulActionNode {
public:
  AdaHandNode(const std::string &name, const BT::NodeConfig &config,
              ada::Ada *robot)
      : BT::StatefulActionNode(name, config), mAda(robot) {}

  static BT::PortsList providedPorts() {
    // Input port for both named preshapes and configurations
    switch (T) {
    case kPRESHAPE:
      // Input named preshape
      return {BT::InputPort<std::string>("preshape")};
    case kCONFIG:
      // Input
      return {BT::InputPort<std::vector<double>>("config")};
    default:
      return {};
    }
  }

  BT::NodeStatus onStart() override {
    BT::Expected<std::string> preshape;
    switch (T) {
    case kOPEN:
      mFuture = mAda->openHand();
      break;
    case kCLOSE:
      mFuture = mAda->closeHand();
      break;
    case kPRESHAPE:
      // Read string named preshape from blackboard
      preshape = getInput<std::string>("preshape");
      if (!preshape)
        return BT::NodeStatus::FAILURE;

      mFuture = mAda->getHand()->executePreshape(preshape.value());
      break;
    default: // case kCONFIG
      // Read string configuration from blackboard
      BT::Expected<std::vector<double>> config =
          getInput<std::vector<double>>("config");
      if (!config)
        return BT::NodeStatus::FAILURE;

      Eigen::VectorXd eConfig = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
          config.value().data(), config.value().size());
      mFuture = mAda->getHand()->executePreshape(eConfig);
    }

    // Do initial check of Future
    return onRunning();
  }

  BT::NodeStatus onRunning() override {
    if (!mFuture.valid())
      return BT::NodeStatus::FAILURE;

    // Check if future is ready
    if (mFuture.wait_for(std::chrono::duration<int, std::milli>(0)) ==
        std::future_status::ready) {
      try {
        mFuture.get();
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return BT::NodeStatus::FAILURE;
      }
      return BT::NodeStatus::SUCCESS;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {
    // Cancel any hand command
    mAda->getHandRobot()->cancelAllCommands();
  }

private:
  ada::Ada *mAda;
  std::future<void> mFuture;
};

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

  // Hand Interface
  factory.registerNodeType<AdaHandNode<kOPEN>>("AdaOpenHand", &robot);
  factory.registerNodeType<AdaHandNode<kCLOSE>>("AdaCloseHand", &robot);
  factory.registerNodeType<AdaHandNode<kPRESHAPE>>("AdaHandPreshape", &robot);
  factory.registerNodeType<AdaHandNode<kCONFIG>>("AdaHandConfig", &robot);
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
