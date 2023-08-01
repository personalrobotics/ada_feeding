# feeding_web_app_ros2_test

This directory contains all the ROS2 nodes that are used to test the web app. This includes: (1) dummy nodes that have the same interface as robot nodes, but actually just sleep; and (2) nodes for the "Test ROS" page of the web app, which demonstrates how to use our ROS helper functions.

## Dependencies
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- Python dependencies:
```
python3 -m pip install numpy pymongo shapely tornado
```

## Usage

To launch the dummy nodes, rosbridge, and web_video_server, run `source install/setup.bash; ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml`.

You can also toggle off certain combinations of dummy nodes with arguments:
- **Don't run motion nodes**: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_motion:=false`
- **Don't run perception nodes**: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_perception:=false`
- **Don't run the RealSense nodes**: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_real_sense:=false`

You can also combine any of the above arguments.
