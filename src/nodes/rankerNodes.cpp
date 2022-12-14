#include "feeding/nodes.hpp"
/**
 * Nodes for ranking/handling DetectionObjects
 **/

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <cmath>

#include <aikido/perception/DetectedObject.hpp>
using aikido::perception::DetectedObject;

namespace feeding {
namespace nodes {

/// Rank by Distance
/// Default: Distance to EE
/// Option: Closest to other object
/// Option: Closest to 3d point
class RankDistance : public BT::SyncActionNode {
public:
  RankDistance(const std::string &name, const BT::NodeConfig &config,
               ada::Ada *robot, ros::NodeHandle *nh)
      : BT::SyncActionNode(name, config), mAda(robot), mNode(nh) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<std::vector<DetectedObject>>("objects"),
            BT::InputPort<DetectedObject>("nearestTo"),
            BT::InputPort<std::vector<double>>("nearestToPos"),
            BT::OutputPort<DetectedObject>("output")};
  }

  BT::NodeStatus tick() override {
    // Pull Params
    auto objectInput = getInput<std::vector<DetectedObject>>("objects");
    if (!objectInput || objectInput.value().size() < 1) {
      return BT::NodeStatus::FAILURE;
    }
    std::vector<DetectedObject> objs = objectInput.value();

    auto objInput = getInput<DetectedObject>("nearestTo");
    auto posInput = getInput<std::vector<double>>("nearestToPos");

    auto distFunc = [](DetectedObject a, DetectedObject b,
                       Eigen::Vector3d target) {
      Eigen::Isometry3d aTransform =
          a.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
      Eigen::Isometry3d bTransform =
          b.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();

      double aDist = (aTransform.translation() - target).norm();
      double bDist = (bTransform.translation() - target).norm();

      return aDist < bDist;
    };
    Eigen::Vector3d target = Eigen::Vector3d::Zero();

    // Sort by Pose Distance
    if (posInput && posInput.value().size() == 3) {
      target << posInput.value()[0], posInput.value()[1], posInput.value()[2];
    }
    // Sort by Object Distance
    else if (objInput) {
      target = objInput.value()
                   .getMetaSkeleton()
                   ->getBodyNode(0)
                   ->getWorldTransform()
                   .translation();
    }
    // Sort by EE Distance
    else {
      target =
          mAda->getEndEffectorBodyNode()->getWorldTransform().translation();
    }

    std::sort(objs.begin(), objs.end(),
              std::bind(distFunc, std::placeholders::_1, std::placeholders::_2,
                        target));
    setOutput<DetectedObject>("output", objs[0]);
    return BT::NodeStatus::SUCCESS;
  }

private:
  ada::Ada *mAda;
  ros::NodeHandle *mNode;
};

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada &robot) {
  factory.registerNodeType<RankDistance>("RankDistance", &robot, &nh);
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding