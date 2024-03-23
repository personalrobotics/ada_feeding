# Ada Feeding Perception
This code performs image segmentation using the [Segment Anything](https://github.com/facebookresearch/segment-anything). It takes an input image and a point as input and generates masks for the regions of interest in the image.

## Setup

See the [`ada_feeding` top-level README for setup instructions](https://github.com/personalrobotics/ada_feeding/blob/ros2-devel/README.md).

For running SegmentAnything test scripts (below), be sure to unzip `test/food_img.zip`.

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

## Food-on-Fork Detection

Our eye-in-hand Food-on-Fork Detection node and training/testing infrastructure was designed to make it easy to substitute and compare other food-on-fork detectors. Below are instructions on how to do so.

1. **Developing a new food-on-fork detector**: Create a subclass of `FoodOnForkDetector` that implements all of the abstractmethods. Note that as of now, a model does not have access to a real-time TF Buffer during test time; hence, **all transforms that the model relies on must be static**.
2. **Gather the dataset**: Because this node uses the eye-in-hand camera, it is sensitive to the relative pose between the camera and the fork. If you are using PRL's robot, [the dataset collected in early 2024](https://drive.google.com/drive/folders/1hNciBOmuHKd67Pw6oAvj_iN_rY1M8ZV0?usp=drive_link) may be sufficient. Otherwise, you should collect your own dataset:
    1. The dataset should consist of a series of ROS2 bags, each recording the following: (a) the aligned depth to color image topic; (b) the color image topic; (c) the camera info topic (we assume it is the same for both); and (d) the TF topic(s).
    2. We recorded three types of bags: (a) bags where the robot was going through the motions of feeding without food on the fork and without the fork nearing a person or plate; (b) the same as above but with food on the fork; and (c) bags where the robot was acquiring and feeding a bite to someone. We used the first two types of bags for training, and the third type of bag for evaluation.
    3. All ROS2 bags should be in the same directory, with a file `bags_metadata.csv` at the top-level of that directory.
    4. `bags_metadata.csv` contains the following columns: `rosbag_name` (str), `time_from_start` (float), `food_on_fork` (0/1), `arm_moving` (0/1). The file only needs rows for timestamps when one or both of the latter columns change; for intermediate timestamps, it is assumed that they stay the same.
    5. To generate `bags_metadata.csv`, we recommend launching RVIZ, adding your depth and/or RGB image topic, and playing back the bag. e.g.,
        1. `ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true`
        2. `ros2 bag play 2024_03_01_two_bites_3 --clock`
        3. Pause and play the rosbag script when food foes on/off the fork, and when the arm starts/stops moving, and populate `bags_metadata.csv` accordingly (elapsed time since bag start should be visible at the bottom of RVIZ2).
3. **Train/test the model on offline data**: We provide a flexible Python script, `food_on_fork_train_test.py`, to train, test, and/or compare one-or-more food-on-fork models. To use it, first ensure you have built and sourced your workspace, and you are in the directory that contains the script (e.g., `cd ~/colcon_ws/src/ada_feeding/ada_feeding_perception/ada_feeding_perception`). To enable flexible use, the script has **many** command-line arguments; we recommend you read their descriptions with `python3 food_on_fork_train_test.py -h`. For reference, we include the command we used to train our model below:
    ```
    python3 food_on_fork_train_test.py --model-classes '{"distance_no_fof_detector_with_filters": "ada_feeding_perception.food_on_fork_detectors.FoodOnForkDistanceToNoFOFDetector"}' --model-kwargs '{"distance_no_fof_detector_with_filters": {"camera_matrix": [614.5933227539062, 0.0, 312.1358947753906, 0.0, 614.6914672851562, 223.70831298828125, 0.0, 0.0, 1.0], "min_distance": 0.001}}' --lower-thresh 0.25 --upper-thresh 0.75 --train-set-size 0.5 --crop-top-left 344 272 --crop-bottom-right 408 336 --depth-min-mm 310 --depth-max-mm 340 --rosbags-select 2024_03_01_no_fof 2024_03_01_no_fof_1 2024_03_01_no_fof_2 2024_03_01_no_fof_3 2024_03_01_no_fof_4 2024_03_01_fof_cantaloupe_1 2024_03_01_fof_cantaloupe_2 2024_03_01_fof_cantaloupe_3 2024_03_01_fof_strawberry_1 2024_03_01_fof_strawberry_2 2024_03_01_fof_strawberry_3 2024_02_29_no_fof 2024_02_29_fof_cantaloupe 2024_02_29_fof_strawberry --seed 42  --temporal-window-size 5 --spatial-num-pixels 10
    ```
Note that we trained our model on data where the fork either had or didn't have food the whole time, and didn't near any objects (e.g., the plate or the user's mouth). (Also, note that not all the above ROS2 bags are necessary; we've trained accurate detectors with half of them.) We then did an offline evaluation of the model on bags of actual feeding data:
    ```
    python3 food_on_fork_train_test.py --model-classes '{"distance_no_fof_detector_with_filters": "ada_feeding_perception.food_on_fork_detectors.FoodOnForkDistanceToNoFOFDetector"}' --model-kwargs '{"distance_no_fof_detector_with_filters": {"camera_matrix": [614.5933227539062, 0.0, 312.1358947753906, 0.0, 614.6914672851562, 223.70831298828125, 0.0, 0.0, 1.0], "min_distance": 0.001}}' --lower-thresh 0.25 --upper-thresh 0.75 --train-set-size 0.5 --crop-top-left 308 248 --crop-bottom-right 436 332 --depth-min-mm 310 --depth-max-mm 340 --rosbags-select 2024_03_01_two_bites 2024_03_01_two_bites_2 2024_03_01_two_bites_3 2024_02_29_two_bites --seed 42  --temporal-window-size 5 --spatial-num-pixels 10 --no-train
    ```
4. **Test the model on online data**: First, copy the parameters you used when training your model, as well as the filename of the saved model, to `config/food_on_fork_detection.yaml`. Re-build and source your workspace. 
    1. **Live Robot**: 
        1. Launch the robot as usual; the `ada_feeding_perception`launchfile will launch food-on-fork detection.
        2. Toggle food-on-fork detection on: `ros2 service call /toggle_food_on_fork_detection std_srvs/srv/SetBool "{data: true}"`
        3. Echo the output of food-on-fork detection: `ros2 topic echo /food_on_fork_detection`
    2. **ROS2 bag data**:
        1. Launch perception: `ros2 launch ada_feeding_perception ada_feeding_perception.launch.py`
        2. Toggle food-on-fork detection on and echo the output of food-on-fork detection, as documented above.
        4. Launch RVIZ and play back a ROS2 bag, as documented above.
