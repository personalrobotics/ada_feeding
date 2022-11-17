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

TODO

## Running the Demo on the JACO 2

### Additional Workspace Setup

1) Build your workspace with `catkin build`
2) Download the checkpoint by going into `src/pytorch_retinanet` and running `load_checkpoint.sh` (or train your own checkpoint)
2) Do the same in `src/bite_selection_package`: run `load_checkpoint.sh` (or train your own checkpoint)
3) Make sure your source `devel/setup.bash` in *every* terminal you use.

###

TODO

## Running with acquisition detection
To run with acquisition detection (and not require manual success/failure input from the supervisor) run
`roslaunch ada_launch default.launch feeding:=true acquisition_detection:=true`

The model files will need to be downloaded by running `ada_demos/feeding/bash_scripts/download_detector_checkpoint.sh`

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
- If you run into internet connection problems, try running `sudo route delete default gw 192.168.1.1`. In general, when running `route -n`, you should see the gateway `192.168.2.1` *above* `192.168.1.1`.

## Safety notes
- The feeding demo has collision boxes for you and your computer, so the robot shouldn't try to hit you usually. But still:
- You can stop Ada's movement by `Ctrl-C`-ing `feeding.launch`.
- **Never use the joystick while the controllers (step 7) are running.** Both will fight over control of Ada and they will not care about collision boxes.
- Be familiar with the location of Ada's on/off-switch :)

## Misc Notes
- 3D models and details for the joule, which mounts the RealSense onto the gen2 arm, can be found [here](https://github.com/ramonidea/wireless-data-transmission/tree/master/wiki).
