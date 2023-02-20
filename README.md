# ADA Feeding Demo

This package demonstrates the use of the [ADA robot](https://github.com/personalrobotics/libada) for robot-assisted feeding.

It uses a [Behavior Tree](https://www.behaviortree.dev/) paradigm (v4.0).

See [trees] for a list of all demos.

See [nodes] for a list of all created CPP nodes.

## Installation

`ada_feeding` is a [ROS1](http://wiki.ros.org/ROS/Installation) catkin package most recently tested on Ubuntu 20.04 LTS (Focal Fossa).

### APT Dependencies
```
DISTRO=noetic
sudo apt install libopencv-dev libblas-dev liblapack-dev libmicrohttpd-dev libeigen3-dev ros-$DISTRO-control-toolbox ros-$DISTRO-ompl ros-$DISTRO-force-torque-sensor-controller ros-$DISTRO-srdfdom python3-wstool ros-$DISTRO-octomap-ros ros-$DISTRO-joint-trajectory-controller ros-$DISTRO-transmission-interface ros-$DISTRO-cv-bridge ros-$DISTRO-image-transport ros-$DISTRO-image-geometry ros-$DISTRO-diagnostic-updater ros-$DISTRO-controller-manager ros-$DISTRO-rviz python-catkin-tools
```

You might also need to install pybind: `pip install pybind11[global]`

You should also symlink `python` to `python3` otherwise some scripts will be unable to find the python binary, causing "No such file or directory" errors when running `roslaunch`:
```
sudo apt install python-is-python3
```

### PRL Git Packages

We can install these all at once with `wstool`.

If running everything in simulation, use `ada-feeding-sim` (which installs fewer packages), otherwise, use `ada-feeding`.

```
$ git clone https://github.com/personalrobotics/pr-rosinstalls.git ~/pr-rosinstalls
$ cd my_catkin_workspace/src
$ wstool init # exclude if already have .rosinstall
$ wstool merge ~/pr-rosinstalls/ada-feeding[-sim].rosinstall
$ wstool up
```

Note that some of the directories installed with the above rosinstall file may have special dependencies that were unmentioned in the README. If you run into errors building/running specific packages, refer to the READMEs of those packages for more details.

### Kinova JACO SDK (Optional: Real Robot Only)

Download the **Gen2 SDK** ZIP file from [Kinova's Website](https://www.kinovarobotics.com/resources). Install the Debian Package inside.

It should be in `Ubuntu/16_04/64 bits/`.

You can install it using `dpkg`, e.g. (for version 6.1.0):
```
sudo dpkg -i KinovaAPi-6.1.0-amd64.deb
```

Note that the current demo has only been tested on the JACO 2.

## Running the Demo in Simulation

1. Ensure that your Workspace is built: `cd <catkin_ws>; catkin build; . devel/setup.bash`
2. Start up ROS and rviz: `roscore` and `roslaunch ada_feeding rviz.launch`
3. Start up simulated perception: `roslaunch ada_feeding perception.launch sim:=true`
4. Run Simulation: `roslaunch ada_feeding feeding.launch sim:=true`. This runs `trees/feeding.xml` by default.
5. Start up an RQT Publisher: `rosrun rqt_publisher rqt_publisher`
6. Set up publications to the following topics:
    1. `/watchdog`: (std_mgs/Bool). A heartbeat that triggers E-Stop if it stops publishing. Publish `True` at 100Hz. **(Note: This will start the demo)**
    2. `/feeding/check_acquire` (std_msgs/Bool). Is checked for True/False after acquisition to determine success.
    3. `/feeding/user_ready` (std_msgs/Bool). Is checked for True for pre-transfer and after transfer to determine when to advance the demo.
    4. `/alexa_msgs` (std_msgs/String). (Mapped from `~food_request`). Is checked pre-acquisition to determine which food type to acquire.

## Running the Demo on the JACO 2

### Additional Workspace Setup

1) Build your workspace with `catkin build`
2) Download the checkpoint by going into `src/pytorch_retinanet` and running `load_checkpoint.sh` (or train your own checkpoint)
2) Do the same in `src/bite_selection_package`: run `load_checkpoint.sh` (or train your own checkpoint)
3) Make sure your source `devel/setup.bash` in *every* terminal you use.

### Demo Run Steps

1. Ensure that your Workspace is built: `cd <catkin_ws>; catkin build; . devel/setup.bash`
2. Start up **ROS** and **Rviz**: `roscore` and `roslaunch ada_feeding rviz.launch`
3. **Turn on and home ADA.** Once the lights on the joystick go solid, home ADA by holding the orange button until the robot stops moving.
4. **Start the Camera**: `ssh nano` (you may need to add `nano` to your `.ssh/config`, this is the Nvidia Jetson Nano on the robot).
    1. Once there, set your ROS Master using `usemaster <hostname>` (e.g. `usemaster weebo` or `usemaster ed209`)
    2. Execute `roslaunch realsense2_camera rs_aligned_depth.launch`  to start streaming RGBD data.
    3. *Note: SSH Key for Nano is available on secrets drive for convenient access*
    4. Check the image stream via Rviz (`/camera/color/image_raw/color`). If some area is too bright and look burnt or saturated, reduce the exposure.
5. **Run F/T Sensor**: `roslaunch forque_sensor_hardware forque.launch` (Optionally add `forque_ip:=<IPv4>` if your Net-FT is on a non-default IP)
6. **Run Face Detection**: `rosrun face_detection face_detection`
7. **(Optional) Run Alexa code**: cd to the `ADA_Talk` directory, and run:
      a) `roslaunch rosbridge_server rosbridge_websocket.launch`
      b) `bst proxy lambda index.js`
8. **Start Demo Code**: `roslaunch ada_feeding feeding.launch sim:=false` (*Note: this should also set `use_forque:=true` and `use_apriltag_calib:=true`*)
9. **Start Perception**: `roslaunch ada_feeding perception.launch`
10. Follows steps 5-6 of "Running the Demo in Simulation" to actually run the demo with `rqt_publisher`.

## Other things to note
- After running the demo one time, the Joystick switches from cartesian control to joint control until you restart Ada.

#### Compilation Troubleshooting

* **DLIB_NO_GUI_SUPPORT**: If you get this error when building `face_detection`: un-comment the `#define` statement in `/usr/include/dlib/config.h`.
* `/usr/include/dlib/opencv/cv_image.h:37:29: error: conversion from ‘const cv::Mat’ to non-scalar type ‘IplImage’ {aka ‘_IplImage’} requested
   37 |             IplImage temp = img;`: If you get this error when building 'face_detection': replace line 37 in `/usr/include/dlib/opencv/cv_image.h` with `IplImage temp = cvIplImage(img);`

##### Additional workspace notes
- There are some repositories that have `ada` in their name but are out of date! Only the repositories in the rosinstall above should be required.
- `openrave` is out of date and not required for this project.
- Whenever you install something to fix dependencies, make sure to _clean_ the affected repositories before you build them again!
- Whenever you run something, make sure to _source the setup.bash_ in the workspace in _every terminal_ you use! We recommend putting it in your `~/.bashrc` file.
- If you have dartsim in the workspace, it might not link to `libnlopt` correctly and you might see an error message when compiling `libada`. When this happens, remove dartsim and install `sudo apt-get libdart6-all-dev`.

## Safety notes
- You can stop Ada's movement by `Ctrl-C`-ing `feeding.launch`.
- **Never use the joystick while the controllers (step 7) are running.** Both will fight over control of Ada and they will not care about collision boxes.
- Be familiar with the location of Ada's on/off-switch :)

## Misc Notes
- 3D models and details for the Jetson Nano, which mounts the RealSense onto the gen2 arm, can be found [here](https://github.com/ramonidea/wireless-data-transmission/tree/master/wiki).

## Shared Autonomy for Plate Locator
1. Turn on ADA and open up `feedingwebapp` in your smartphone (follow instructions [here](https://github.com/personalrobotics/feeding_web_interface/tree/2022_revamp/feedingwebapp)).
2. Start moving the robot clicking on directional buttons in the app until you get an alert that a partial plate has been located.
3. Open up a terminal window and run `find_image_center.py` in `catkin_ws` directory. It will print out translational (x, y, z) and rotational (pitch, yaw, roll) position values for desired full plate center.
4. Go to the `full_plate_locator.xml` tree file and plug in those position values in the robot node or port of `AdaPlanToPoseOffset`. Be sure to save the file with new changes.
5. Run `roslaunch ada_feeding full_plate_locator.launch` in `catkin_ws` to launch the `full_plate_locator.xml` tree file for moving the robot and locating the full plate.
6. If you want to start feeding now, just run `roslaunch ada_feeding feeding.launch sim:=false` in `catkin_ws` to run the demo code.

## Full Autonomy for Plate Locator
1. Add your desired position values of translation (x, y, z) and rotation (pitch, yaw, roll) for the robot node or port of `AdaPlanToPoseOffset` in the `partial_plate_locator.xml` tree file. Be sure to save the file with new changes.
2. Run `roslaunch ada_feeding partial_plate_locator.launch` in `catkin_ws` to launch the `partial_plate_locator.xml` tree file for systematically moving the robot with measured spaces. The file will stop running after locating the partial plate. 
3. Open up a terminal window and run `find_image_center.py` in `catkin_ws` directory. It will print out translational (x, y, z) and rotational (pitch, yaw, roll) position values for desired full plate center.
4. Go to the `full_plate_locator.xml` tree file and plug in those position values in the robot node or port of `AdaPlanToPoseOffset`. Be sure to save the file with new changes.
5. Run `roslaunch ada_feeding full_plate_locator.launch` in `catkin_ws` to launch the `full_plate_locator.xml` tree file for moving the robot and locating the full plate.
6. If you want to start feeding now, just run `roslaunch ada_feeding feeding.launch sim:=false` in `catkin_ws` to run the demo code.


