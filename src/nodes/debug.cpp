#include "feeding/nodes.hpp"
/**
 * Test and debugging BT nodes
 **/

#include <behaviortree_cpp_v3/behavior_tree.h>
#include <iostream>

namespace feeding {
namespace nodes {

BT::NodeStatus Success() { return BT::NodeStatus::SUCCESS; }

BT::NodeStatus Failure() { return BT::NodeStatus::FAILURE; }

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory,
                          ros::NodeHandle & /*nh*/, ada::Ada & /*robot*/) {
  factory.registerSimpleAction("Success", std::bind(Success));
  factory.registerSimpleAction("Failure", std::bind(Failure));
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
