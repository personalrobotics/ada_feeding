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

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada &robot) {}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding