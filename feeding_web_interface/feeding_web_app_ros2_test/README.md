# feeding_web_app_ros2_test

This directory contains all the ROS2 nodes that are used to test the web app. This includes: (1) dummy nodes that have the same interface as robot nodes, but actually just sleep; and (2) nodes for the "Test ROS" page of the web app, which demonstrates how to use our ROS helper functions.

## Dependencies
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- Python dependencies:
```
python3 -m pip install numpy shapely
```