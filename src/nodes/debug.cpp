#include "feeding/nodes.hpp"
/**
 * Test and debugging BT nodes
 **/

#include <behaviortree_cpp/behavior_tree.h>
#include <iostream>

namespace feeding {
namespace nodes {

BT::NodeStatus Success() { return BT::NodeStatus::SUCCESS; }

BT::NodeStatus Failure() { return BT::NodeStatus::FAILURE; }

// Basic Pause/Debugging Console
BT::NodeStatus DebugNode() {
  std::cout << "=== Debugging Console" << std::endl;
  std::cout << "f - return FAILURE" << std::endl;
  std::cout << "s - return SUCCESS" << std::endl;
  std::cout << "r - return RUNNING" << std::endl;

  char choice = '\0';
  const std::set<char> valid = {'f', 's', 'r'};
  while (valid.find(choice) == valid.end()) {
    std::cout << "> ";
    choice = std::cin.get();
    std::cin.ignore(256, '\n');
  }

  std::string key = "";
  BT::Expected<std::string> value;
  switch (choice) {
  case 's':
    std::cout << "=== Returning SUCCESS" << std::endl;
    return BT::NodeStatus::SUCCESS;
  case 'r':
    std::cout << "=== Returning RUNNING" << std::endl;
    return BT::NodeStatus::RUNNING;
  default:
    std::cout << "=== Returning FAILURE" << std::endl;
    return BT::NodeStatus::FAILURE;
  }
}

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory,
                          ros::NodeHandle & /*nh*/, ada::Ada & /*robot*/) {
  factory.registerSimpleAction("Success", std::bind(Success));
  factory.registerSimpleAction("Failure", std::bind(Failure));

  // Note: Simple action nodes will throw error on RUNNING
  factory.registerSimpleCondition("Debug", std::bind(DebugNode));
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
