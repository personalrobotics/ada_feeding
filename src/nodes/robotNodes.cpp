#include "feeding/nodes.hpp"
/**
 * Nodes for controlling ADA
 **/

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <functional>
#include <iostream>
#include <vector>

#include <aikido/rviz.hpp>
extern aikido::rviz::InteractiveMarkerViewerPtr gMarkerViewer;

#include <aikido/common/util.hpp>
using aikido::common::FuzzyZero;

namespace feeding {
namespace nodes {

//// Planning Nodes
enum PlanToNodeType { kOFFSET, kCONFIG, kTSR, kPOSEOFFSET };
template <PlanToNodeType T> class PlanToNode : public BT::StatefulActionNode {
public:
  PlanToNode(const std::string &name, const BT::NodeConfig &config,
             ada::Ada *robot)
      : BT::StatefulActionNode(name, config), mAda(robot) {}

  static BT::PortsList providedPorts() {
    // Input port for the offset vector
    // Output port for the traj
    BT::PortsList ret = {
        BT::InputPort<bool>("worldCollision"),
        BT::OutputPort<aikido::trajectory::TrajectoryPtr>("traj")};

    switch (T) {
    case kPOSEOFFSET:
      ret.insert(BT::InputPort<std::vector<double>>("rotation"));
      [[fallthrough]];
    case kOFFSET:
      ret.insert(BT::InputPort<std::vector<double>>("offset"));
      break;
    case kTSR:
      ret.insert(BT::InputPort<std::vector<double>>("orig_pos"));
      ret.insert(BT::InputPort<std::vector<double>>("orig_quat"));
      ret.insert(BT::InputPort<std::vector<double>>("pos"));
      ret.insert(BT::InputPort<std::vector<double>>("quat"));
      ret.insert(BT::InputPort<std::vector<double>>("bounds"));
      ret.insert(BT::InputPort<std::vector<double>>("lbounds"));
      ret.insert(BT::InputPort<bool>("viz"));
      break;
    case kCONFIG:
    default:
      ret.insert(BT::InputPort<std::vector<double>>("config"));
    }

    return ret;
  }

  BT::NodeStatus onStart() override {
    auto wc = getInput<bool>("worldCollision");
    bool worldCollision = wc ? wc.value() : false;
    auto constraint = worldCollision
                          ? mAda->getArm()->getWorldCollisionConstraint()
                          : mAda->getArm()->getSelfCollisionConstraint();

    bool success = false;
    switch (T) {
    case kPOSEOFFSET:
      success = planToPoseOffset(constraint);
      break;
    case kOFFSET:
      success = planToOffset(constraint);
      break;
    case kTSR:
      success = planToTSR(constraint);
      break;
    case kCONFIG:
    default:
      success = planToConfig(constraint);
    }
    if (!success)
      return BT::NodeStatus::FAILURE;

    // Do initial check of Future
    return onRunning();
  }

  BT::NodeStatus onRunning() override {
    if (!mFuture.valid())
      return BT::NodeStatus::FAILURE;

    // Check if future is ready
    if (mFuture.wait_for(std::chrono::duration<int, std::milli>(0)) ==
        std::future_status::ready) {
      aikido::trajectory::TrajectoryPtr traj = nullptr;
      try {
        traj = mFuture.get();
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        setOutput<aikido::trajectory::TrajectoryPtr>("traj", nullptr);
        return BT::NodeStatus::FAILURE;
      }
      setOutput<aikido::trajectory::TrajectoryPtr>("traj", traj);
      return traj ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {
    // Future will go out of scope
    // We can ignore the computation result.
  }

private:
  // Individual planning functions
  bool planToPoseOffset(aikido::constraint::TestablePtr constraint) {
    auto offset = getInput<std::vector<double>>("offset");
    if (!offset || offset.value().size() != 3) {
      return false;
    }
    Eigen::Vector3d eOffset(offset.value().data());
    auto rotation = getInput<std::vector<double>>("rotation");
    if (!rotation || rotation.value().size() != 3) {
      return false;
    }
    Eigen::Vector3d eRotation(rotation.value().data());

    mFuture = std::async(
        std::launch::async,
        [this](Eigen::Vector3d off, Eigen::Vector3d rot,
               aikido::constraint::TestablePtr testable) {
          return mAda->getArm()->planToPoseOffset(
              mAda->getEndEffectorBodyNode()->getName(), off, rot, testable);
        },
        eOffset, eRotation, constraint);
    return true;
  }

  bool planToOffset(aikido::constraint::TestablePtr constraint) {
    auto offset = getInput<std::vector<double>>("offset");
    if (!offset || offset.value().size() != 3) {
      return false;
    }
    Eigen::Vector3d eOffset(offset.value().data());

    mFuture = std::async(
        std::launch::async,
        [this](Eigen::Vector3d off, aikido::constraint::TestablePtr testable) {
          return mAda->getArm()->planToOffset(
              mAda->getEndEffectorBodyNode()->getName(), off, testable);
        },
        eOffset, constraint);
    return true;
  }

  bool planToConfig(aikido::constraint::TestablePtr constraint) {
    auto config = getInput<std::vector<double>>("config");
    if (!config) {
      return false;
    }
    Eigen::VectorXd eConfig = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        config.value().data(), config.value().size());

    mFuture = std::async(
        std::launch::async,
        [this](Eigen::VectorXd conf, aikido::constraint::TestablePtr testable) {
          return mAda->getArm()->planToConfiguration(conf, testable);
        },
        eConfig, constraint);

    return true;
  }

  bool planToTSR(aikido::constraint::TestablePtr constraint) {
    auto tsr = std::make_shared<aikido::constraint::dart::TSR>();

    // TSR Origin (w) relative to world origin (0)
    auto wpos = getInput<std::vector<double>>("orig_pos");
    auto wquat = getInput<std::vector<double>>("orig_quat");
    tsr->mT0_w = Eigen::Isometry3d::Identity();
    if (wpos) {
      if (wpos.value().size() != 3)
        return false;
      tsr->mT0_w.translation() = Eigen::Vector3d(wpos.value().data());
    }
    if (wquat) {
      if (wquat.value().size() != 4)
        return false;
      tsr->mT0_w.linear() =
          Eigen::Quaterniond(wquat.value()[0], wquat.value()[1],
                             wquat.value()[2], wquat.value()[3])
              .toRotationMatrix();
    }

    // End effector (e) relative to TSR origin (w)
    auto epos = getInput<std::vector<double>>("pos");
    auto equat = getInput<std::vector<double>>("quat");
    tsr->mTw_e = Eigen::Isometry3d::Identity();
    if (epos) {
      if (epos.value().size() != 3)
        return false;
      tsr->mTw_e.translation() = Eigen::Vector3d(epos.value().data());
    }
    if (equat) {
      if (equat.value().size() != 4)
        return false;
      tsr->mTw_e.linear() =
          Eigen::Quaterniond(equat.value()[0], equat.value()[1],
                             equat.value()[2], equat.value()[3])
              .toRotationMatrix();
    }

    // TSR Bounds in TSR frame (w)
    auto hbounds = getInput<std::vector<double>>("bounds");
    auto lbounds = getInput<std::vector<double>>("lbounds");
    tsr->mBw = Eigen::Matrix<double, 6, 2>::Zero();
    if (hbounds) {
      if (hbounds.value().size() != 6)
        return false;
      Eigen::Vector6d bounds(hbounds.value().data());
      tsr->mBw.col(1) = bounds;
      tsr->mBw.col(0) = -bounds;
    }
    if (lbounds) {
      if (lbounds.value().size() != 6)
        return false;
      Eigen::Vector6d bounds(lbounds.value().data());
      tsr->mBw.col(0) = bounds;
    }

    auto viz = getInput<bool>("viz");
    if (viz && viz.value()) {
      ROS_WARN_STREAM("Visualizing Marker");
      gMarkerViewer->addTSRMarker(*tsr);
    }

    mFuture = std::async(
        std::launch::async,
        [this](aikido::constraint::dart::TSRPtr tsr,
               aikido::constraint::TestablePtr testable) {
          return mAda->getArm()->planToTSR(
              mAda->getEndEffectorBodyNode()->getName(), tsr, testable);
        },
        tsr, constraint);

    return true;
  }

  // Members
  ada::Ada *mAda;
  std::future<aikido::trajectory::TrajectoryPtr> mFuture;
};

//// Execution Nodes
class ExecuteTrajNode : public BT::StatefulActionNode {
public:
  ExecuteTrajNode(const std::string &name, const BT::NodeConfig &config,
                  ada::Ada *robot)
      : BT::StatefulActionNode(name, config), mAda(robot) {}

  static BT::PortsList providedPorts() {
    // Input port for traj
    BT::PortsList ret = {
        BT::InputPort<aikido::trajectory::TrajectoryPtr>("traj")};

    return ret;
  }

  BT::NodeStatus onStart() override {
    auto traj = getInput<aikido::trajectory::TrajectoryPtr>("traj");
    if (!traj || !traj.value())
      return BT::NodeStatus::FAILURE;

    // TODO: this will draw from the end of the arm
    // This will NOT draw from the gripper
    // Works with the robot metaskeleton,
    // but if you don't use a metaskeleton clone,
    // it will race with the Ada thread.
    auto ms = mAda->getArm()->getMetaSkeletonClone();
    gMarkerViewer->addTrajectoryMarker(
        traj.value(), ms, *(ms->getBodyNode(ms->getNumBodyNodes() - 1)),
        Eigen::Vector4d(0.0, 0.75, 0.5, 0.8));

    mFuture = mAda->getArm()->executeTrajectory(traj.value());

    // Do initial check of Future
    return onRunning();
  }

  BT::NodeStatus onRunning() override {
    if (!mFuture.valid())
      return BT::NodeStatus::FAILURE;

    // Check if future is ready
    if (mFuture.wait_for(std::chrono::duration<int, std::milli>(0)) ==
        std::future_status::ready) {
      gMarkerViewer->clearTrajectoryMarkers();
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
    // Cancel execution
    mAda->getArm()->cancelAllCommands();
    gMarkerViewer->clearTrajectoryMarkers();
  }

private:
  ada::Ada *mAda;
  std::future<void> mFuture;
};

/// Read EE Pose
BT::NodeStatus getEEPose(BT::TreeNode &self, ada::Ada &robot) {
  Eigen::Isometry3d pose = robot.getEndEffectorBodyNode()->getTransform();
  Eigen::Vector3d ePos = pose.translation();
  self.setOutput<std::vector<double>>(
      "pos", std::vector<double>(ePos.data(), ePos.data() + ePos.size()));

  Eigen::Quaterniond eQuat(pose.linear());
  std::vector<double> quat{eQuat.w(), eQuat.x(), eQuat.y(), eQuat.z()};
  self.setOutput<std::vector<double>>("quat", quat);
  return BT::NodeStatus::SUCCESS;
}

// Read Configuration
BT::NodeStatus getConfig(BT::TreeNode &self, ada::Ada &robot) {
  auto sArmOnly = self.getInput<bool>("armOnly");
  bool armOnly = (sArmOnly) ? sArmOnly.value() : true;
  Eigen::VectorXd eConfig =
      (armOnly) ? robot.getArm()->getMetaSkeleton()->getPositions()
                : robot.getMetaSkeleton()->getPositions();
  self.setOutput<std::vector<double>>(
      "target",
      std::vector<double>(eConfig.data(), eConfig.data() + eConfig.size()));
  return BT::NodeStatus::SUCCESS;
}

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory,
                          ros::NodeHandle & /*&nh */, ada::Ada &robot) {
  // Planning Functions
  factory.registerNodeType<PlanToNode<kPOSEOFFSET>>("AdaPlanToPoseOffset",
                                                    &robot);
  factory.registerNodeType<PlanToNode<kOFFSET>>("AdaPlanToOffset", &robot);
  factory.registerNodeType<PlanToNode<kCONFIG>>("AdaPlanToConfig", &robot);
  factory.registerNodeType<PlanToNode<kTSR>>("AdaPlanToPose", &robot);

  // Trajectory Execution
  factory.registerNodeType<ExecuteTrajNode>("AdaExecuteTrajectory", &robot);

  // Get EE Pose
  factory.registerSimpleAction(
      "AdaGetEEPose",
      std::bind(getEEPose, std::placeholders::_1, std::ref(robot)),
      {BT::OutputPort<std::vector<double>>("pos"),
       BT::OutputPort<std::vector<double>>("quat")});

  // Get Joint Configuration
  factory.registerSimpleAction(
      "AdaGetConfig",
      std::bind(getConfig, std::placeholders::_1, std::ref(robot)),
      {BT::InputPort<bool>("armOnly"),
       BT::OutputPort<std::vector<double>>("target")});
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
