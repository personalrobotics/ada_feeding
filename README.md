# ada_feeding

This tutorial focuses on the assistive feeding demo on an Ubuntu machine.

## Packages to install

### Ubuntu Repos

#### Standard (Non-Docker) Installation

Assuming you already have `ros-$DISTRO-desktop-full` installed (as per [here](http://wiki.ros.org/ROS/Installation)) and the latest version of [DART](https://dartsim.github.io/install_dart_on_ubuntu.html), here are the additional dependencies required from other repositories:

```
DISTRO=noetic
sudo apt install libopencv-dev libblas-dev liblapack-dev libmicrohttpd-dev libeigen3-dev ros-$DISTRO-control-toolbox ros-$DISTRO-ompl ros-$DISTRO-force-torque-sensor-controller ros-$DISTRO-srdfdom python3-wstool ros-$DISTRO-octomap-ros ros-$DISTRO-joint-trajectory-controller ros-$DISTRO-transmission-interface ros-$DISTRO-cv-bridge ros-$DISTRO-image-transport ros-$DISTRO-image-geometry ros-$DISTRO-diagnostic-updater ros-$DISTRO-controller-manager ros-$DISTRO-rviz python-catkin-tools
```
Replace `noetic` with the appropriate ROS version: `melodic` on 18.04 (Focal) and `kinetic` on 16.04 (Xenial).

You might also need to install pybind: `pip install pybind11[global]`

If you cannot install `python-catkin-tools` you may have to use this workaround to install directly from the source repo, for more details see [this issue](https://github.com/catkin/catkin_tools/issues/594):
```
pip3 install -U "git+https://github.com/catkin/catkin_tools.git#egg=catkin_tools"
```

If you are on 20.04 (Focal), you should also symlink `python` to `python3` otherwise some scripts will be unable to find the python binary, causing "No such file or directory" errors when running `roslaunch`:
```
sudo apt install python-is-python3
```

#### Docker Installation

Docker containers are a useful way to isolate the environments used to develop and run code. They can be particularly useful to avoid clashes between different versions or distributions of libraries (for example, if you have an older version of ROS on your host computer but want to run ADA code with a newer version), and to enable code to easily work across computers. We have verified that the below instructions, and the included Dockerfile, work for running the demo in simulation.

1. Configure the Docker PPA following the [latest instructions](https://docs.docker.com/engine/install/ubuntu/) (i.e., the instructions under `Install using the repository > Set up the repository`).
2. Install the Docker packages.
```
sudo apt-get update
sudo apt-get install docker-ce
```
3. Start the Docker service: `sudo service docker start`
4. Create a new catkin workspace for use within your Docker container, and follow the below instructions in "PRL Git Packages" to load its packages.
5. Navigate to `ada_feeding/docker`.
6. Build the docker container using `sudo docker build -t ada-sim .`
7. Enable the Docker container to render GUIs on the host computer with:
```
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
```
8. Run the Docker container using `sudo docker run --rm --network host -v my_catkin_workspace:/workspace -v $XSOCK:$XSOCK:rw -v $XAUTH:$XAUTH:rw -e XAUTHORITY=${XAUTH} -e DISPLAY  -it ada-sim`

You are good to go! Note that any files created within the Docker container that are not within a mounted volume will be deleted once you exit the container (the relevant mounted volume in the above command is your catkin workspace). Further note that the best practice is to have a separate workspace for use within the Docker container, and to only build/source that workspace from within the container (to avoid clashes with ROS versions, library paths, etc.). Finally, note that steps 7-8 above have to be re-run whenever you want to enter your container.

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

Run the following commands from your ROS workspace:

1. `catkin build`
1. `source devel/setup.bash`
1. `roscore`
1. `rviz`
1. `roslaunch ada_launch simulation.launch` (will put 2 simulated *cantaloupe* on the plate)
1. `roslaunch ada_feeding feeding.launch` (will quit after writing ROS parameters)
1. `cd my_catkin_workspace/devel/bin/` and `./feeding`
1.  In RViz, subscribe to the topic `feeding/update/InteractiveMarkers` to actually see the robot.

## Running the Demo on the JACO 2

### Additional Workspace Setup

1) Build your workspace with `catkin build`
2) Download the checkpoint by going into `src/pytorch_retinanet` and running `load_checkpoint.sh` (or train your own checkpoint)
2) Do the same in `src/bite_selection_package`: run `load_checkpoint.sh` (or train your own checkpoint)
3) Make sure your source `devel/setup.bash` in *every* terminal you use.

###

1) Start `roscore`, `rviz`
2) Turn on ADA
3) Once the lights on the joystick go solid, home ADA by holding the orange button until the robot stops moving.
4) `ssh nano` (you may need to add `nano` to your `.ssh/config`, this is the Jetson on the robot). Once there, set your ROS Master using `uselovelace`, `useweebo`, or `useweebowired` (or set your ROS_MASTER_URI manually), execute `./run_camera.sh` to start streaming RGBD data.
   * You may have to adjust the camera exposure, depending on the lighting condition. Either run `run_adjust_camera_daylight.sh` or `run_adjust_camera_all.sh` after running `run_camera.sh`. Check the image stream via rviz, by adding the image topic `/camera/color/image_raw/color`. If some area is too bright and look burnt or saturated, reduce the exposure.
6) `roslaunch forque_sensor_hardware forque.launch` (Optionally add `forque_ip:=<IPv4>` if your Net-FT is on a non-default IP)
6) `rosrun face_detection face_detection`
7) `roslaunch ada_launch default.launch feeding:=true detector:=spanet`
   * Optionally, run with `perception:=false` if you want to run perception manually in another terminal.
   * To run perception manually in another terminal: `rosrun food_detector run_perception_module.py --demo-type spanet`
8) `roslaunch ada_feeding feeding.launch` (will quit after writing ROS parameters)
   * Optionally run `roslaunch ada_feeding data_collection.launch` after `feeding.launch` if you're doing data collection.
9) `cd ~/Workspace/ada_ws/devel/bin/` and `./feeding -af`
    * `-a`: specified that this is the real robot, and not a simulation
    * `-f`: enables the force/torque sensor on the real robot (**REQUIRED when picking up food for safety**)
    * `-c`: causes demo to proceed without the user pressing \[ENTER\] between steps
    * `-d`: Select demo to run.
10) Advance the demo by pressing Return or terminate nicely by pressing `n` and then return

## Running with acquisition detection
To run with acquisition detection (and not require manual success/failure input from the supervisor) run
`roslaunch ada_launch default.launch feeding:=true acquisition_detection:=true`

The model files will need to be downloaded by running `ada_demos/feeding/bash_scripts/download_detector_checkpoint.sh`

## Other things to note
- In step 7, `feeding:=true` is responsible for loading the robot with camera and forque
- Step 8 loads ros parameters for the feeding executable
- When the demo exits, it shuts down some controllers that were started. If it crashes, you'll need to restart the controller node (step 7)
- If launching the controllers (step 7) doesn't work properly, chances are a `JacoHardware` node didn't exit cleanly. I recommend checking for that by turning every ros node off and running `ps -aux | grep jaco` to see if a process is still running and preventing your controllers from starting properly.
- After running the demo one time, the Joystick switches from cartesian control to joint control until you restart Ada.

#### Compilation Troubleshooting

* **DLIB_NO_GUI_SUPPORT**: If you get this error when building `face_detection`: un-comment the `#define` statement in `/usr/include/dlib/config.h`.

##### Additional workspace notes
- There are some repositories that have `ada` in their name but are out of date! Only the repositories in the rosinstall above should be required.
- `openrave` is out of date and not required for this project.
- Whenever you install something to fix dependencies, make sure to _clean_ the affected repositories before you build them again!
- Whenever you run something, make sure to _source the setup.bash_ in the workspace in _every terminal_ you use! We recommend putting it in your `~/.bashrc` file.
- If you have dartsim in the workspace, it might not link to `libnlopt` correctly and you might see an error message when compiling `libada`. When this happens, remove dartsim and install `sudo apt-get libdart6-all-dev`.
- If you run into internet connection problems, try running `sudo route delete default gw 192.168.1.1`. In general, when running `route -n`, you should see the gateway `192.168.2.1` *above* `192.168.1.1`.

## Safety notes
- The feeding demo has collision boxes for you and your computer, so the robot shouldn't try to hit you usually. But still:
- You can stop Ada's movement by `Ctrl-C`-ing the controller node started in step 7.
- **Never use the joystick while the controllers (step 7) are running.** Both will fight over control of Ada and they will not care about collision boxes.
- Be familiar with the location of Ada's on/off-switch :)
