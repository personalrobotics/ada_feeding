# Feeding Web Interface

This repository contains code for the feeding web app. The app itself is in `feedingwebapp`. ROS nodes used to test the app are in `feeding_web_app_ros2_test`, and messages for those ROS nodes are in `feeding_web_app_ros2_msgs`. Each directory contains its own README.

## ROS Dependencies
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- [PRL fork of rosbridge_suite](https://github.com/personalrobotics/rosbridge_suite). This fork enables rosbridge_suite to communicate with ROS2 actions.
- [ada_feeding (branch: `ros2-devel`)](https://github.com/personalrobotics/ada_feeding/tree/ros2-devel).
- [web_video_server (branch: `ros2`)](https://github.com/RobotWebTools/web_video_server/tree/ros2)
    - Dependency: `ros-humble-async-web-server-cpp`
    - Dependency: `ros-humble-vision-opencv`
    
All these repositories of `feeding_web_interface`, `ada_feeding`, `async_web_server_cpp`, `vision_opencv`, `web_video_server`, and `PRL fork of rosbridge_suite` as mentioned above should be downloaded using `git clone ...` in the "src" folder inside the ROS2 workspace. Please follow the [Ubuntu (Debian) tutorial](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) for ROS2 installation in Ubuntu 22, which will automatically lead to creation of "src" folder in ROS2 workspace. To access hidden `.env` file in Ubuntu to change the debug flag's value, press Ctrl + H in the `feeding_web_interface` folder. 
