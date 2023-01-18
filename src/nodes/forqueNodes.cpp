#include "feeding/nodes.hpp"
/**
 * Nodes for interfacing with the FTThresholdHelper
 **/

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <functional>
#include <iostream>
#include <vector>

#include <feeding/FTThresholdHelper.hpp>

namespace feeding {
namespace nodes {
static std::shared_ptr<FTThresholdHelper> sFTThreshHelper = nullptr;

class SetFTThresh : public BT::StatefulActionNode {

public:
  SetFTThresh(const std::string &name, const BT::NodeConfig &config)
      : BT::StatefulActionNode(name, config) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::string>("preset"),
            BT::InputPort<double>("force"), BT::InputPort<double>("torque"),
            BT::InputPort<bool>("retare")};
  }

  BT::NodeStatus onStart() override {
    // Read Params
    auto retareInput = getInput<bool>("retare");
    bool retare = retareInput ? retareInput.value() : false;

    auto preset = getInput<std::string>("preset");
    if (preset) {
      mFuture = std::async(
          std::launch::async,
          [&](std::string preset, bool retare) {
            return sFTThreshHelper->setThresholds(preset, retare);
          },
          preset.value(), retare);
    } else {
      auto forceInput = getInput<double>("force");
      auto torqueInput = getInput<double>("torque");
      if (!forceInput || !torqueInput) {
        return BT::NodeStatus::FAILURE;
      }
      mFuture = std::async(
          std::launch::async,
          [&](double force, double torque, bool retare) {
            return sFTThreshHelper->setThresholds(force, torque, retare);
          },
          forceInput.value(), torqueInput.value(), retare);
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
      bool success = false;
      try {
        success = mFuture.get();
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return BT::NodeStatus::FAILURE;
      }
      return success ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {
    // Nothing to do
  }

private:
  std::future<bool> mFuture;
};

BT::NodeStatus FTStartDataCollect(BT::TreeNode &self) {
  auto numInput = self.getInput<int>("num_points");
  size_t num = numInput ? (size_t)numInput.value() : 0;
  auto ret = sFTThreshHelper->startDataCollection(num);
  return ret ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

BT::NodeStatus FTStopDataCollect(BT::TreeNode & /*self*/) {
  sFTThreshHelper->stopDataCollection();
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus FTGetData(BT::TreeNode &self) {
  if (!sFTThreshHelper->isDataCollectionFinished())
    return BT::NodeStatus::FAILURE;

  self.setOutput<std::vector<double>>("data", sFTThreshHelper->getData());
  return BT::NodeStatus::SUCCESS;
}

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada & /*robot*/) {
  bool isSim = nh.param("sim", true);
  bool use_forque = nh.param("use_forque", true);
  if (!use_forque && !isSim)
    ROS_ERROR_STREAM("FORQUE IS DISABLED! DO NOT USE FOR FEEDING!");
  sFTThreshHelper =
      std::make_shared<FTThresholdHelper>((!isSim && use_forque), nh);
  sFTThreshHelper->init();

  factory.registerNodeType<SetFTThresh>("SetFTThreshold");

  // F/T Data Collection
  factory.registerSimpleAction(
      "FTStartDataCollect",
      std::bind(FTStartDataCollect, std::placeholders::_1),
      {BT::InputPort<int>("num_points")});
  factory.registerSimpleAction("FTStopDataCollect", FTStopDataCollect);
  factory.registerSimpleAction("FTGetData",
                               std::bind(FTGetData, std::placeholders::_1),
                               {BT::OutputPort<std::vector<double>>("data")});
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
