# ada_feeding

This README is the definitive source for downloading, installing, and running the Personal Robotics Lab's robot-assisted feeding software. This code has been run and tested on machines running **Ubuntu 22.04** and ROS2 Humble.

## Setup

### Setup (Robot Software)

1. [Install ROS2 Humble](https://docs.ros.org/en/humble/Installation.html) (binary packages are recommended over building from source), [configure your environment](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html), and [create a workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html#).
    1. **NOTE**: In the remainder of the instructions, replace `~\colcon_ws` with the path to your workspace.
2. Configure [`pr-rosinstalls`](https://github.com/personalrobotics/pr-rosinstalls) in order to download all necessary repositories.

        cd ~
        git clone https://github.com/personalrobotics/pr-rosinstalls.git

3. Pull all the repositories, with the correct branch. Replace `<https/ssh>` with one of the options, depending on your authentication method.

        sudo apt install python3-wstool # if not already installed
        cd ~/colcon_ws/src
        wstool init
        wstool merge ~/pr-rosinstalls/ada-feeding.<https/ssh>.rosinstall
        wstool up

4. Configure [`rosdep`](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html):

        sudo apt install python3-rosdep # if not already installed
        sudo rosdep init # if this is the first time using rosdep

5. Install [`rosdep`](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html) dependencies:

        rosdep update
        cd ~/colcon_ws
        rosdep install --from-paths src -y --ignore-src --as-root=pip:false --filter-for-installers pip

   If you have sudo access and are setting up a computer for the first time, run the last command without `--filter-for-installers pip` to install all apt dependencies that require sudo access.

7. Install non-`rosdep` dependencies:
    - Install SegmentAnythingModel: `python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git`
    - Install EfficientSAM: `python3 -m pip install git+https://github.com/yformer/EfficientSAM.git`
    - Upgrade `transforms3d`, since [the release on Ubuntu packages is outdated](https://github.com/matthew-brett/transforms3d/issues/65): `python3 -m pip install transforms3d -U`
    - [`pyrealsense2` is not released for ARM systems](https://github.com/IntelRealSense/librealsense/issues/6449#issuecomment-650784066), so ARM users will have to [build from source](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md#building-from-source). You may have to add the `-DPYTHON_EXECUTABLE=/usr/bin/python3` flag to the `cmake` command. When running `sudo make install`, pay close attention to which path `pyrealsense2` is installed to and add *that path* to the `PYTHONPATH` -- it should be `/use/local/lib` but may be `/usr/local/OFF`.
8. Install the JACO SDK (real robot only). All SDKs are listed [here](https://www.kinovarobotics.com/resources?r=79301&s); PRL currently uses the [Gen2 SDK v1.5.1](https://drive.google.com/file/d/1UEQAow0XLcVcPCeQfHK9ERBihOCclkJ9/view). Note that although the latest version of that SDK is for Ubuntu 16.04, it still works on Ubuntu 22.04 (only for x86 systems, not ARM system).
9. Build your workspace:

        cd ~/colcon_ws
        colcon build --symlink-install # if sim-only, add '--packages-skip ada_hardware'

### Setup (Web App)

1. Install the Node Version Manager (nvm): https://github.com/nvm-sh/nvm?tab=readme-ov-file#install--update-script
2. Install and use NodeJS 21:

        nvm install 21
        nvm use 21

3. Make Node available to all users, including root:

        sudo ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/node" "/usr/local/bin/node"
        sudo ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/npm" "/usr/local/bin/npm"
        sudo ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/npx" "/usr/local/bin/npx"

4. Install `serve` and `pm2` globally. Root access is necessary for `serve` so it can access port 80.

        sudo npm install -g serve
        npm install -g pm2@latest

5. Install the web app dependencies. (Note: there will be some vulnerabilities in dependencies. That is okay, since )

        cd ~/colcon_ws/src/feeding_web_interface/feedingwebapp
        npm install --legacy-peer-deps
        npx playwright install

6. (Optional; this should already be configured on PRL computers) To access the web app on a device other than the one hosting it, enable the desired port for HTTP access: https://www.digitalocean.com/community/tutorials/ufw-essentials-common-firewall-rules-and-commands#allow-all-incoming-http-port-80


## Running the Software

We use the convenience script `start.py` to launch the software. This script has several command-line arguements, which can be seen by passing the `-h` flag when running the script.

### **Recommended** Option A: Run the Web App with the Real Robot

This option starts the web app and the real robot code, and can be used to test the entire system. This will by default start the web app on port `80`, and requires `sudo` access.

```
cd ~/colcon_wsUbuntu (Debian packages)
python3 src/ada_feeding/start.py
```

In a browser, access `127.0.0.1` (if on the same device serving the web app), or the IP address of the device hosting the web app (if on a different device connected to the same network). You should now be able to run the system! Note that upon startup, the watchdog is in a failing state until the e-stop is clicked exactly once, allowing the system to verify that it is connected and working.

To close, run `python3 src/ada_feeding/start.py -c`


### Option B: Running Web App With the Mock Robot

This option starts the web app, runs dummy nodes for perception, runs the **real** robot motion code, but runs a mock robot in MoveIt. This is useful for testing robot motion code in simulation before moving onto the real robot. This will start the web app on port `3000` and does not require `sudo` access.

**NOTE**: Before running `mock`, it is recommended to disable the Octomap by [changing the name of `default_sensor/image_topic` to a topic that is not published](https://github.com/personalrobotics/ada_ros2/blob/e5256bfc89c358cb71699c6be95e78bf846eed63/ada_moveit/config/sensors_3d.yaml#L7). Be sure to re-build after the change.

```
cd ~/colcon_ws
python3 src/ada_feeding/start.py --sim mock
```

In a browser, access `127.0.0.1:3000` (if on the same device serving the web app), or the IP address of the device hosting the web app at port `3000` (if on a different device connected to the same network). You should now be able to run the system!

To close, run `python3 src/ada_feeding/start.py --sim mock -c`

### Option C: Running Web App With All Dummy Nodes

This option starts the web app, and runs dummy nodes for all perception and robot motion code. This is useful to test the web app in isolation. This will start the web app on port `3000` and does not require `sudo` access.

```
cd ~/colcon_ws
python3 src/ada_feeding/start.py --sim dummy
```

In a browser, access `127.0.0.1:3000` (if on the same device serving the web app), or the IP address of the device hosting the web app at port `3000` (if on a different device connected to the same network). You should now be able to run the system!

To close, run `python3 src/ada_feeding/start.py --sim dummy -c`

## Troubleshooting

- **Something is not working, what do I do?**: All our code runs in `screen` sessions, so the first step is to attach to individual screen sessions to identify where the issue is. Run `screen -ls` to list all screens, run `screen -r <name>` to attach to a screen session, and `Ctrl-A d` to detach from the screen session. An introduction to `screen` can be found [here](https://astrobiomike.github.io/unix/screen-intro).
- **The watchdog is not recognizing my initial e-stop click**: Verify the e-stop is plugged in, and that any other processes accessing the microphone (e.g., Sound Settings) are closed. Run `alsamixer` to see if your user account has access to it. If you do not see sound levels in the terminal, try `sudo alsamixer`. If that works, give your user account permission to access sound: `sudo setfacl -m u:$USER:rw /dev/snd/*`
- **The watchdog is failing due to the F/T sensor**: First, check whether the force-torque sensor is publishing: `ros2 topic echo /wireless_ft/ftSensor1`. If not, the issue is in the `ft` screen (one potential issue is that the alias `ft-sensor` does not point to the right IP address for the force-torque sensor, in which case you should pass the IP address in using the `host` parameter). If so, check the timestamp of the published F/T messages compared to the current time. If it is off, the problem is that NTP got disabled on the force-torque sensor. You have to use `minicom` to re-enable NTP (see [here](https://github.com/personalrobotics/pr_docs/wiki/ADA) for PRL-specific instructions). 
- **I get the error `cannot use destroyable because destruction was requested`**: Upgrade to the latest version of`rclpy`.
- **I get the error `bad option: --env-file=.env` when launching the WebRTC Signalling Server**: You are using an old version of Node; upgrade it to 21.
