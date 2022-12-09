#include "feeding/nodes.hpp"
/**
 * Nodes to manipulate the Aikido World
 **/

#include <Eigen/Core>
#include <aikido/io/util.hpp>
#include <behaviortree_cpp/behavior_tree.h>
#include <iostream>

#include <aikido/rviz.hpp>
extern aikido::rviz::InteractiveMarkerViewerPtr gMarkerViewer;

namespace feeding {
namespace nodes {

// Add/Update Skeleton to world
BT::NodeStatus AddUpdateSkeleton(BT::TreeNode &self, ada::Ada &robot) {
  auto world = robot.getWorld();
  const auto resourceRetriever =
      std::make_shared<aikido::io::CatkinResourceRetriever>();

  // Get Pose relative to world origin
  auto pos = self.getInput<std::vector<double>>("pos");
  std::vector<double> vpos{0.0, 0.0, 0.0};
  if (pos && pos.value().size() == 3) {
    vpos = pos.value();
  }
  Eigen::Vector3d ePos(vpos.data());

  auto quat = self.getInput<std::vector<double>>("quat");
  std::vector<double> vquat{1.0, 0.0, 0.0, 0.0};
  if (quat && quat.value().size() == 4) {
    vquat = quat.value();
  }
  Eigen::Quaterniond eQuat(vquat[0], vquat[1], vquat[2], vquat[3]);
  eQuat.normalize();

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = eQuat.toRotationMatrix();
  pose.translation() = ePos;

  // Determine if add or update operation
  auto urdfUri = self.getInput<std::string>("urdfUri");
  auto name = self.getInput<std::string>("skelName");

  // Cannot operate on ADA
  if (name && name.value() == robot.getRootSkeleton()->getName()) {
    ROS_WARN_STREAM("Cannot update ADA itself!");
    return BT::NodeStatus::FAILURE;
  }

  if (!urdfUri) {
    // Pose-only update operation
    if (!name) {
      // No name or URDF provided
      return BT::NodeStatus::FAILURE;
    }
    auto skeleton = world->getSkeleton(name.value());
    if (!skeleton) {
      // Name provided has no skeleton in world
      return BT::NodeStatus::FAILURE;
    }

    // Update skeleton root free joint
    auto freeJoint =
        dynamic_cast<dart::dynamics::FreeJoint *>(skeleton->getRootJoint());

    if (!freeJoint) {
      ROS_WARN_STREAM(
          "Cannot cast root free joint for skeleton: " << name.value());
      return BT::NodeStatus::FAILURE;
    }

    freeJoint->setTransform(pose);
  } else {
    // Load Skeleton
    auto skeleton =
        loadSkeletonFromURDF(resourceRetriever, urdfUri.value(), pose);
    if (name) {
      skeleton->setName(name.value());
    }

    // Skeleton Update Operation, remove old skeleton
    auto oldSkeleton = world->getSkeleton(skeleton->getName());
    if (oldSkeleton) {
      world->removeSkeleton(oldSkeleton);
    }

    // Add new skeleton
    world->addSkeleton(skeleton);
  }

  return BT::NodeStatus::SUCCESS;
}

// Remove Skeleton From World
BT::NodeStatus RemoveSkeleton(BT::TreeNode &self, ada::Ada &robot) {
  auto world = robot.getWorld();
  auto name = self.getInput<std::string>("skelName");
  if (!name) {
    return BT::NodeStatus::FAILURE;
  }

  // Cannot remove robot from world
  if (name.value() == robot.getRootSkeleton()->getName()) {
    ROS_WARN_STREAM("Cannot remove ADA from its own world!");
    return BT::NodeStatus::FAILURE;
  }

  auto skeleton = world->getSkeleton(name.value());
  if (skeleton) {
    world->removeSkeleton(skeleton);
  }
  return BT::NodeStatus::SUCCESS;
}

// Clear Frame and Trajectory Markers
BT::NodeStatus ClearFrameMarkers(BT::TreeNode & /* self */) {
  gMarkerViewer->clearFrameMarkers();
  return BT::NodeStatus::SUCCESS;
}
BT::NodeStatus ClearTrajMarkers(BT::TreeNode & /* self */) {
  gMarkerViewer->clearTrajectoryMarkers();
  return BT::NodeStatus::SUCCESS;
}

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory,
                          ros::NodeHandle & /*nh*/, ada::Ada &robot) {
  factory.registerSimpleAction(
      "WorldAddUpdate",
      std::bind(AddUpdateSkeleton, std::placeholders::_1, std::ref(robot)),
      {BT::InputPort<std::string>("urdfUri"),
       BT::InputPort<std::string>("skelName"),
       BT::InputPort<std::vector<double>>("pos"),
       BT::InputPort<std::vector<double>>("quat")});

  factory.registerSimpleAction(
      "WorldRemove",
      std::bind(RemoveSkeleton, std::placeholders::_1, std::ref(robot)),
      {BT::InputPort<std::string>("skelName")});

  factory.registerSimpleAction("WorldClearFrames", ClearFrameMarkers);
  factory.registerSimpleAction("WorldClearTraj", ClearTrajMarkers);
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding