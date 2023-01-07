#include "feeding/nodes.hpp"
/**
 * Test and debugging BT nodes
 **/

#include <behaviortree_cpp/behavior_tree.h>
#include <iostream>

// For system_async
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

/// Codes for async starting and killing processes
pid_t system_async(const char *command, int *infp = NULL, int *outfp = NULL) {
  int p_stdin[2];
  int p_stdout[2];
  pid_t pid;

  if (pipe(p_stdin) == -1)
    return -1;

  if (pipe(p_stdout) == -1) {
    close(p_stdin[0]);
    close(p_stdin[1]);
    return -1;
  }

  pid = fork();

  if (pid < 0) {
    close(p_stdin[0]);
    close(p_stdin[1]);
    close(p_stdout[0]);
    close(p_stdout[1]);
    return pid;
  } else if (pid == 0) {
    close(p_stdin[1]);
    dup2(p_stdin[0], 0);
    close(p_stdout[0]);
    dup2(p_stdout[1], 1);
    dup2(open("/dev/null", O_RDONLY), 2);
    /// Close all other descriptors for the safety sake.
    for (int i = 3; i < 4096; ++i)
      ::close(i);

    setsid();
    execl("/bin/sh", "sh", "-c", command, NULL);
    _exit(1);
  }

  close(p_stdin[0]);
  close(p_stdout[1]);

  if (infp == NULL) {
    close(p_stdin[1]);
  } else {
    *infp = p_stdin[1];
  }

  if (outfp == NULL) {
    close(p_stdout[0]);
  } else {
    *outfp = p_stdout[0];
  }

  return pid;
}

BT::NodeStatus RunProcess(BT::TreeNode &self) {
  auto cmdInput = self.getInput<std::string>("cmd");
  if (!cmdInput)
    return BT::NodeStatus::FAILURE;
  int pid = system_async(cmdInput.value().c_str());

  self.setOutput<int>("pid", pid);
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus RunProcessBlocked(BT::TreeNode &self) {
  auto cmdInput = self.getInput<std::string>("cmd");
  if (!cmdInput)
    return BT::NodeStatus::FAILURE;
  ROS_WARN_STREAM("Running System Command: " << cmdInput.value());
  int ret = std::system(cmdInput.value().c_str());
  return (ret == 0) ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

BT::NodeStatus KillProcess(BT::TreeNode &self) {
  auto pidInput = self.getInput<int>("pid");
  if (!pidInput)
    return BT::NodeStatus::FAILURE;
  auto codeInput = self.getInput<int>("code");
  if (kill(pidInput.value(), codeInput ? codeInput.value() : SIGINT)) {
    ROS_WARN_STREAM(strerror(errno));
    return BT::NodeStatus::FAILURE;
  }
  return BT::NodeStatus::SUCCESS;
}

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory,
                          ros::NodeHandle & /*nh*/, ada::Ada & /*robot*/) {
  factory.registerSimpleAction("Success", std::bind(Success));
  factory.registerSimpleAction("Failure", std::bind(Failure));

  // Note: Simple action nodes will throw error on RUNNING
  factory.registerSimpleCondition("Debug", std::bind(DebugNode));

  // Process Code
  factory.registerSimpleAction(
      "DebugRunProcessBlocked",
      std::bind(RunProcessBlocked, std::placeholders::_1),
      {BT::InputPort<std::string>("cmd")});
  factory.registerSimpleAction(
      "DebugRunProcess", std::bind(RunProcess, std::placeholders::_1),
      {BT::InputPort<std::string>("cmd"), BT::OutputPort<int>("pid")});
  factory.registerSimpleAction(
      "DebugKillProcess", std::bind(RunProcess, std::placeholders::_1),
      {BT::InputPort<int>("pid"), BT::InputPort<int>("code")});
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
