#!/bin/bash
source ~/ros2_humble/install/local_setup.bash
source ~/Workspace/camera_ros2/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch realsense2_camera rs_launch.py rgb_camera.profile:=640,480,15 depth_module.profile:=640,480,15 align_depth.enable:=true initial_reset:=true publish_tf:=false


