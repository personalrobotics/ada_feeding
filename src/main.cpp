/**
 * Entry point to the ADA Feeding System
 **/
#include <behaviortree_cpp_v3/bt_factory.h>
#include <feeding/nodes.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include <libada/Ada.hpp>

int main(int argc, char** argv)
{
  // Initialize ROS node
  ros::init(argc, argv, "feeding");
  std::shared_ptr<ros::NodeHandle> nh = std::make_shared<ros::NodeHandle>("~");
  ros::AsyncSpinner spinner(2); // 2 threads
  spinner.start();

  // Construct ADA Robot Object
  ROS_INFO("Initializing ADA...");
  bool isSim = nh.param("sim", true);
  ada::Ada robot(isSim);

  // Start Visualization
  ROS_INFO("Starting RViz Visualization...");
  aikido::rviz::InteractiveMarkerViewer viewer(
      nh.param("visualization/topicName", "dart_markers/feeding"),
      nh.param("visualization/topicName", "dart_markers/feeding"),
      robot.getWorld());
  viewer->setAutoUpdate(true);

  // Register Behavior Tree Nodes
  ROS_INFO("Initializing Behavior Tree...");
  BT::BehaviorTreeFactory factory;
  feeding::registerNodes(factory, nh, robot);

  // Create tree
  std::string defaultTree = ros::package::getPath("ada_feeding") + "/trees/default.xml";
  std::string treeFile = nh.param("treeFile", defaultTree);
  auto tree = factory.createTreeFromFile(treeFile);
  // This logger prints state changes on console
  BT::StdCoutLogger logger_cout(tree);

  // Run Tree until SUCCESS or FAILURE
  // Then repeat
  while(ros::ok()) {
    auto status = BT::NodeStatus::RUNNING;
    while(status == BT::NodeStatus::RUNNING) {
      status = tree.tickRoot();
      tree.sleep(std::chrono::milliseconds(10));
    }
  }
  return 0;
}
