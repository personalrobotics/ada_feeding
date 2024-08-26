#!/bin/bash
source ~/ros2_humble/install/local_setup.bash
source ~/Workspace/camera_ros2/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch nano_bridge nano_bridge_sender.launch.xml


