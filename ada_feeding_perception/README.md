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

### Dependencies
- [joblib](https://joblib.readthedocs.io/en/stable/installing.html#using-pip)

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

## Food on Fork
The Food on Fork node gets launched automatically when the Ada Feeding perception is launched. As such, after 
launching the perception nodes, one can perform `ros2 topic echo /food_on_fork` to be able to listen to the 
confidence being published onto the topic. Note that 0.0 indicates `no_food` and 1.0 indicates `food` on the fork. 
Additionally, the web application has a toggle button that can get triggered on or off to be able to listen to the  
`/food_on_fork` topic. More documentation for the webapp can be found on its repository. 

There are a couple models that can be used to determine Food on Fork. They are as follows:
> Note that "frustum" is a 3D space around the fork

- Logistic Regression
  - This model uses a single feature (number of pixels within the specified frustum). Based on that, it outputs a confidence on whether or not there is a presence of food on fork.
- Categorical Naive Bayes 
  - Within the Categorical Naive Bayes model, there are two ones, namely `categorical_naive_bayes_depth_8-30-23` (model A) and `categorical_naive_bayes_aligned_depth_10-17-23` (model B). 
    - Model A uses raw depth images that are already cropped to a specified frustum. And, it treats each "voxel" in the depth image as a feature by itself. So, in essence there are over 10K features on which the Categorical NB is trained on.
    - On the otherhand, Model B uses algined_depth images that are also cropped to a specified frustum (Currently, Food On Fork node uses this model!). The benefit of using this model over others are because: 
      - Using/listening to raw depth images causes issues listening to some of the other topics, which other features 
        (such as Face Detection) are reliant on.
      - Logistic Regression only considered the number of voxels within the frustum, which was eliminating 
        information that we can use from being aware of the location of each of those voxels (in other words, 
        considering each of the voxel as a feature by itself, as opposed to just one feature as the count).

### Training Food on Fork Logisitic Regression model
- Navigate to `food_on_fork_logistic_reg_training.py` within `/ada_feeding_perception` package.
- Make sure to update the `csv_to_read_from` with an updated training set. Note that the current training set is located in `ada_feeding_perception/datasets`.
  - Make sure to change the name of the model by updating `model_save_filename` variable in this file
- Then, run `food_on_fork_logistic_reg_training.py`, making sure its dependencies are installed
- After the completion of training, be sure to navigate to `food_on_fork.yaml` and change the model location so that the correct model is being used.

### Training Food on Fork Categorical Naive Bayes model (that uses aligned_depth instead of raw depth)
- Change directory into `ada_feeding_perception/ada_feeding_perception`.
- Make sure to load the dataset! Since the dataset contains actual aligned depth images, it becomes too large to be 
  committed onto GitHub. As such, it has been uploaded onto the drive. Make sure to download the dataset of your choice and use it as indicated below.
  - [Dataset0](https://drive.google.com/file/d/19yZhHcmpUmAlnM40e2sVdyXHtxWq2bF7/view?usp=sharing), [Dataset1]
    (https://drive.google.com/file/d/1K5xbgm77mS_4Ya-mtGnMXiT0_UbEnH4k/view?usp=sharing), [Dataset2](https://drive.
    google.com/file/d/1SkTP9uE4GOVBodpWMAOPiFG-hlUcR7xq/view?usp=sharing): Note that all three of these datasets are using the same data and have 80/20 train-test split. The difference is that each of them have different images in train/test datasets.
  - Make sure to simply download the dataset into the folder `ada_feeding_perception/ada_feeding_perception`. Please don't unzip the file because the code unzips it at runtime.
- To train CategoricalNB, there are a few required arguments:
  - Suppose you want to use 80/20 train-test split and check the accuracy of the model:
    - `python food_on_fork_categorical_naive_bayes_training.py --use_entire_dataset_bool "False" --data_file_zip 
      "<Absolute Location of zip file for data>"`
      - Make sure to specify the correct zipfile and the boolean argument will be false since we want to only train on the trainset and check its accuracy on the testset.
  - Suppose you want to train and generate a `.pkl` model:
    - `python food_on_fork_categorical_naive_bayes_training.py --use_entire_dataset_bool "<Bool>" --data_file_zip 
      "<Absolute Location of zip file for data>" --model_save_file "<Absolute Location with name for model>"`
    - This will train on the entire dataset and create a model based on that.
  - Additionally, note that you can run `python food_on_fork_categorical_naive_bayes_training.py --help` and it will 
    provide you with some cues as to what are the variables needed
- Make sure to install all dependencies and run the python file. After running, based on the method executed, there maybe a new model created. Be sure to update the `food_on_fork.yaml` file's model location variable to use the correct model.

### Training Food on Fork Categorical Naive Bayes model (that uses raw depth images)
- Follow the same steps as you do in Training using `aligned_depth`, but you will need to load different datasets. 
  They are linked below:
  - [Dataset0](https://drive.google.com/file/d/1KuZonyrz4440pHjgPvTqeinP3aju0xTK/view?usp=sharing)
  - [Dataset1](https://drive.google.com/file/d/1XEIx9CipqyAJqEuqukIb9eiTW1ee8Nyu/view?usp=sharing)
  - [Dataset2](https://drive.google.com/file/d/16Gbn360WE5RXOgbfESxNhWbb9SsEgdC7/view?usp=sharing)