#ifndef FEEDING_NODES_HPP_
#define FEEDING_NODES_HPP_

/**
 * Functions for manipulating and registering BT Nodes
 *
 * Each compilation unit should register its nodes with:
 * static_block { feeding::registerNodeFn(RegistrationFn); }
 **/

#include <behaviortree_cpp/bt_factory.h>
#include <feeding/static_block.hpp>
#include <functional>
#include <libada/Ada.hpp>
#include <ros/ros.h>

namespace feeding {

/// Registration Function List
typedef std::function<void(BT::BehaviorTreeFactory &, ros::NodeHandle &,
                           ada::Ada &)>
    RegistrationFn;

/// Add registration functions to be called by registerNodes().
/// Should be called from static_block{}
/// in each compilation unit with BT nodes.
void registerNodeFn(RegistrationFn Fn);

// Register all BT Nodes to provided factory
// Nodes may use robot / nh for initialization.
void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                   ada::Ada &robot);

} // end namespace feeding

#endif
