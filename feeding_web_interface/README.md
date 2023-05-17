# Feeding Web Interface

This repository contains code for the feeding web app. The app itself is in `feedingwebapp`. ROS nodes used to test the app are in `feeding_web_app_ros2_test`, and messages for those ROS nodes are in `feeding_web_app_ros2_msgs`. Each directory contains its own README.

## ROS Dependencies
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- [PRL fork of rosbridge_suite](https://github.com/personalrobotics/rosbridge_suite). This fork enables rosbridge_suite to communicate with ROS2 actions.
- [ada_feeding (branch: `ros2-devel`)](https://github.com/personalrobotics/ada_feeding/tree/ros2-devel).
- [web_video_server (branch: `ros2`)](https://github.com/RobotWebTools/web_video_server/tree/ros2)
    - Dependency: [async_web_server_cpp (branch: `ros2-develop`)](https://github.com/fkie/async_web_server_cpp)
    - Dependency: [vision_opencv (branch: `humble` or your ROS2 version)](https://github.com/ros-perception/vision_opencv/tree/humble)
