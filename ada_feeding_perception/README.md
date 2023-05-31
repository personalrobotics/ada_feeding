# Ada Feeding Perception
This code performs image segmentation using the [Segment Anything](https://github.com/facebookresearch/segment-anything). It takes an input image and a point as input and generates masks for the regions of interest in the image.

## Installation
1. Clone this directory into the `src` folder of your ROS2 workspace.
2. Install the dependencies:
```
chmod +x install.sh
./install.sh
```

## Usage

1. Build your workspace: `colcon build`
2. Source your workspace: `source install/setup.bash`
3. Run the action servers: `ros2 launch ada_feeding_perception ada_feeding_perception_launch.xml`
4. Test it:
    1. Individual action servers: **TODO**
    2. With the web app:
        1. Launch the motion nodes:
            1. Dummy nodes: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_real_sense:=false run_perception:=false`
            2. Real nodes: `ros2 launch ada_feeding ada_feeding_launch.xml`
        2. Launch the RealSense node:
            1. Dummy nodes: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_motion:=false run_perception:=false`
            2. Real nodes: **TODO**
        3. Launch the web app ([instructions here](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp))

## Running `test_sam.py`

`test_sam.py` is a convenience script provided to ensure that SegmentAnything runs on your computer. It is designed to run as a standalone script. As such, it does not utilize any of the helper functions contained in the `ada_feeding_perception` module (which must be built in a ROS2 workspace). Hence, it cannot be used to test any aspect of the ROS2 action server; it can only be used to test Segment Anything.

To run the `test_sam.py` with an example image and input points, use the following command:

```
python3 test/image_segmentation.py --input_image <example_image.jpg> --input_point [x,y]
```
Here are some examples. Before running them, unzip `test/food_img.zip`.
```
python3 test/test_sam.py --input_image test/food_img/sete_00100.jpg --input_point [800,500]
python3 test/test_sam.py --input_image test/food_img/sete_00100.jpg --input_point [1050,400]
python3 test/test_sam.py --input_image test/food_img/sete_00100.jpg --input_point [1200,400]

python3 test/test_sam.py --input_image test/food_img/setf_00113.jpg --input_point [1050,400]
python3 test/test_sam.py --input_image test/food_img/setf_00113.jpg --input_point [1300,500]
python3 test/test_sam.py --input_image test/food_img/setf_00113.jpg --input_point [1300,800]
```