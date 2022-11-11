/**
 * Entry point to the ADA Feeding System
 **/
#include <aikido/rviz.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/bt_cout_logger.h>
#include <feeding/nodes.hpp>
#include <libada/Ada.hpp>
#include <ros/package.h>
#include <ros/ros.h>

int main(int argc, char **argv) {
  // Initialize ROS node
  ros::init(argc, argv, "feeding");
  ros::NodeHandle nh("~");
  ros::AsyncSpinner spinner(2); // 2 threads
  spinner.start();

  // Construct ADA Robot Object
  ROS_INFO("Initializing ADA...");
  bool isSim = nh.param("sim", true);
  ada::Ada robot(isSim);

  // Start Visualization
  ROS_INFO("Starting RViz Visualization...");
  std::string vizTopic =
      nh.param<std::string>("visualization/topicName", "dart_markers/feeding");
  std::string vizBaseFrame =
      nh.param<std::string>("visualization/baseFrameName", "map");
  aikido::rviz::InteractiveMarkerViewer viewer(vizTopic, vizBaseFrame,
                                               robot.getWorld());
  viewer.setAutoUpdate(true);

  // Register Behavior Tree Nodes
  ROS_INFO("Initializing Behavior Tree...");
  BT::BehaviorTreeFactory factory;
  feeding::registerNodes(factory, nh, robot);

  // Create tree
  std::string defaultTree =
      ros::package::getPath("ada_feeding") + "/trees/default.xml";
  std::string treeFile = nh.param<std::string>("treeFile", defaultTree);
  auto tree = factory.createTreeFromFile(treeFile);
  // This logger prints state changes on console
  BT::StdCoutLogger logger_cout(tree);

  // Run Tree until SUCCESS or FAILURE
  // Then repeat
  bool autoRestart = nh.param("autoRestart", false);
  ros::Rate rate(100); // ROS Rate at 100Hz (10ms)
  while (ros::ok()) {
    auto status = BT::NodeStatus::RUNNING;
    while (status == BT::NodeStatus::RUNNING) {
      status = tree.tickOnce();
      rate.sleep();
    }

    // User input to confirm demo restart
    if (!autoRestart) {
      std::string decision = "";
      while (decision != "y" && decision != "n" && decision != "Y" &&
             decision != "N") {
        std::cout << "Tree complete (" << status << "). Restart (Y/n)? ";
        decision = std::cin.get();
        std::cin.ignore();
      }
      if (decision == "n" || decision == "N")
        break;
    }
  }
  return 0;
}
