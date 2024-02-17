# feeding_web_app_ros2_test

This directory contains all the ROS2 nodes that are used to test the web app. This includes: (1) dummy nodes that have the same interface as robot nodes, but actually just sleep; and (2) nodes for the "Test ROS" page of the web app, which demonstrates how to use our ROS helper functions.

## Setup

See the [`ada_feeding` top-level README for setup instructions](https://github.com/personalrobotics/ada_feeding/blob/ros2-devel/README.md).

## Usage

To launch the dummy nodes, rosbridge, and web_video_server, run `source install/setup.bash; ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml`.

You can also toggle off certain combinations of dummy nodes with arguments:
- **Don't run motion nodes**: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_motion:=false`
- **Don't run the face detection node**: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml :run_face_detection=false`
- **Don't run the food detection node**: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml :run_food_detection=false`
- **Don't run the RealSense node**: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_real_sense:=false`

You can also combine any of the above arguments.

## Simulating an AcquireFood Goal

If you launch the code in `--sim mock` (see [here](https://github.com/personalrobotics/ada_feeding/blob/ros2-devel/README.md)), using bite selection through the app should properly call AcquireFood, and you should be able to see the acquisition action in RVIZ (it is recommended to add an `Axes` visualization to RVIZ for the `food` frame to see the perceived top-center and orientation of the detected food item). However, this approach has two downsides:
1. The detected food mask is a dummy mask, not a real output of SegmentAnything.
2. Each time you use the app to invoke AcquireFood, the mask and depth will be slightly different, which is a challenge for repeatability.

To address these issues, we have pickled goal request(s) from the app to the AcquireFood action. These goal request(s) were Segmented by the actual `SegmentFromPoint` node with the dummy RGB and depth images (found in `./data`). To invoke bite acquisition with this static goal request, do the following:
1. `cd ~/colcon_ws`
2. `python3 src/ada_feeding/start-py --sim mock`. See [here](https://github.com/personalrobotics/ada_feeding/blob/ros2-devel/README.md) for more info.
3. `ros2 launch feeding_web_app_ros2_test feeding_dummy_acquirefood_launch.py`. This will have the robot move above the plate, and then invoke `AcquireFood` for the stored goal request.

There are 6 pickled goal requests in `./data`. You can modify which gets run through a launch argument to `feeding_dummy_acquirefood_launch.py`. Be sure to also change the images published by the DummyRealSense node to correspond to the new pickled goal request; these can be changed by launch argument to `feeding_web_app_dummy_nodes_launch.xml`.

To pickle more goal_requests, modify `ada_feeding/config/` to [pass the `pickle_goal_path` parameter to the AcquireFood tree](https://github.com/personalrobotics/ada_feeding/blob/f889fe44351ec552e945ba028d4928826ee03710/ada_feeding/config/ada_feeding_action_servers_default.yaml#L54). That should be a full path to where you want to store the pickle. Then, run the **dummy RealSense node**, the **real** perception nodes, run the web app, do bite selection for a bite, and select a mask. That mask will be stored in a pickle.
