# Ada Feeding Perception
This code performs image segmentation using the [Segment Anything](https://github.com/facebookresearch/segment-anything). It takes an input image and a point as input and generates masks for the regions of interest in the image.

## Installation
1. Clone this directory into the `src` folder of your ROS2 workspace.
2. Install the Python dependencies:
```
source install.sh
```
3. Install the system dependencies:
```
sudo apt install ros-humble-image-transport ros-humble-compressed-image-transport
```

For testing, be sure to unzip `test/food_img.zip`.

## Usage

1. Build your workspace: `colcon build`
2. Source your workspace: `source install/setup.bash`
3. Run the perception nodes: `ros2 launch ada_feeding_perception ada_feeding_perception.launch.py`
4. Launch the motion nodes:
    1. Dummy nodes: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_real_sense:=false run_face_detection:=false run_food_detection:=false`
    2. Real nodes: `ros2 launch ada_feeding ada_feeding_launch.xml`
5. Launch the RealSense node:
    1. Dummy nodes: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_motion:=false run_face_detection:=false run_food_detection:=false`
        1. NOTE: SegmentFromPoint will no longer work with only the dummy RealSense node, since the dummy RealSense only publishes a color image, whereas SegmentFromPoint also expects a depth image.
    2. Real nodes:
        1. SSH into the `nano` user of `nano`: On a pre-configured lab computer, this should be `ssh nano`. Else, look here for [the IP address of nano](https://github.com/personalrobotics/pr_docs/wiki/Networking-and-SSH-Information).
        2. On `nano`:
            1. `ros2config` (this takes several seconds)
            2. `ros2 launch realsense2_camera rs_launch.py rgb_camera.profile:='640,480,30' depth_module.profile:='640,480,30' align_depth.enable:='true' initial_reset:='true'`. See here for [a complete list of params](https://github.com/IntelRealSense/realsense-ros/blob/ros2-development/realsense2_camera/launch/rs_launch.py). **Note that `nano` must be connected to the same WiFi network as the computer running any ros2 nodes that subscribe to it.**
        3. (To visualize the camera stream, run `ros2 run rviz2 rviz2`)
6. Launch the web app ([instructions here](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp))

## Food Segmentation

Food segmentation currently uses SegmentAnything's base model (`vit_b`) and has the user specify a point on the food they want to segment. It then returns multiple contender masks.

### Benchmarking Food Segmentation Speed

Below are rough numbers that were taken by running the food segmentation action server and accessing the web app on the same machine, starting the timer when we click a point on the web app, and ending the timer when the masks render in the web app. In other words, it includes all delays but web latency.
- On LoveLace's GPU, it takes ~0.6s.
- On LoveLace's CPU, it takes ~4s.
- On an Ubuntu VM (4 core, 8GB of RAM), it takes ~10s.

### Testing Food Segmentation

There are several options for testing food segmentation:

#### Option A: Testing SegmentAnything without ROS

A `test_sam.py` convenience script is provided to run images through SegmentAnything without needing ROS. This script can be useful to ensure that SegmentAnything is working on your computer, or to benchmark how long SegmentAnything takes without additional delays.

To run this script, do:
- `cd ada_feeding_perception`
- `python3 test_sam.py --input_image ../test/food_img/sete_00100.jpg --input_point [800,500]`

See `config/test_segment_from_point.yaml` for other sample images and points. Note that you have to unzip `test/food_img.zip` to run this.

#### Option B: Interactively Testing the ROS Action Server

**NOTE**: On some machines Option B does not work (more generally, matplotlib interactive graphics don't work).

We have provided a ROS node that displays the live image stream from a topic, let's users click on it, and sends that point click to the SegmentFromPoint action server.

Run this script with: `ros2 launch ada_feeding_perception test_food_segmentation_launch.xml`

Note that because we are rendering a live image stream in matplotlib, this script can be very resource intensive. As an example, on one non-GPU machine it **slowed SegmentAnything down by 10x** (since so many resources were going to rendering the video's live stream).

#### Option C: Testing the ROS Action Server on Saved Images

Option C is now **DEPRECATED** because SegmentFromPoint requires aligned depth images, and we don't have aligned depth images for any of the offline images. The below instructions are kept only for legacy purposes.

To facilitate the testing of a large number of stored images, we have provided a script that reads in a list of images and points and sends then to the SegmentFromPoint action server one-at-a-time, saving the results.

To run this script, change the `mode` parameter in `config/test_segment_from_point.yaml` to `offline`. Then run `ros2 launch ada_feeding_perception test_food_segmentation_launch.xml`.

This script should save the output images in `<path/to/your/workspace>/install/ada_feeding_perception/share/ada_feeding_perception/test_img/output`.

#### Option D: Testing Using the Web App

Launch the web app along with all the other nodes (real or dummy) as documented in *"Usage"* above. Then, use the web app as expected, and the food segmentation should be done by the action server.

## Parameters

### `segment_from_point`
- `model_name` (string, required): the name of the SegmentAnything checkpoint to use. You can see the options [here](https://github.com/facebookresearch/segment-anything#model-checkpoints).
- `model_base_url` (string, required): The base URL where the model is hosted. Again, this came from the links [here](https://github.com/facebookresearch/segment-anything#model-checkpoints).
- `n_contender_masks` (int, optional, default 3): The number of contender masks to return.

### `test_segment_from_point`
- `mode` (string, required): Either `offline` or `online`. In `offline` it segments a list of images stored on file and points that are passed in as parameters. In `online`, it displays a live stream and lets users click a point to specify what point to seed the segmentation with.
- `action_server_name` (string, required): The name of the action server to call.
- `image_topic` (string, required): The name of the image topic that the action server subscribes to.
- `offline.save_dir` (string, required for offline): Where, relative to `install/ada_feeding_perception/share/ada_feeding_perception` to save the output images.
- `offline.sleep_time` (float, required): How long to sleep after publishing an image and before calling the action server. This is necessary because the action server performs segmentation on the last received image.
- `offline.images` (list of strings, required): The paths, relative to `install/ada_feeding_perception/share/ada_feeding_perception`, to the images to test.
- `offline.point_xs` (list of ints, required): The x-coordinates of the seed points. Must be the same length as `offline.images`.
- `offline.point_ys` (list of ints, required): The y-coordinates of the seed points. Must be the same length as `offline.images`.
