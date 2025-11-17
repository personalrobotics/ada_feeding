# ada_feeding

This README is the definitive source for downloading, installing, and running the Personal Robotics Lab's robot-assisted feeding software. This code has been run and tested on machines running **Ubuntu 22.04** and ROS2 Humble. The project is designed to be run within a shared Docker container.

This guide is in two parts:

1. **Administrator Setup:** A one time setup on a new lab machine.
2. **New Researcher Workflow:** The day-to-day workflow for researchers.

## **Part 1: Administrator Setup (One-Time)**

This must be performed once by a lab admin on any new host machine.

### **1\. Prerequisites (Host Setup)**

1. **Install Docker:**  
```
   sudo apt install docker.io
```

2. **Add Admin to Docker Group:** This is required to run Docker without `sudo`.  
```
   sudo usermod \-aG docker $USER
```
   **IMPORTANT:** You must **log out and log back in** for this change to take effect.  

3. **Create New User Accounts:** For each new researcher, create a non-admin account and add them to the `docker` group so they can run the shared image.
```
   # Create the user (you will be prompted for a password)
   sudo adduser <new_username>

   # Add the user to the docker group
   sudo usermod -aG docker <new_username>
```

### **2\. Get Project Files (For Building)**

The admin only needs this repo to get the `Dockerfile`.
**Note:** The Articutool project files are currently on feature branches. This step pulls the specific branch needed for the build.

```
# Clone the ada_feeding repo and check out the integration branch
git clone --branch jjaime2/articutool-integration https://github.com/personalrobotics/ada_feeding.git
cd ada_feeding
```

### **3\. Download Kinova SDK**

The Docker build requires the Kinova Gen2 SDK.
1. Download `PS 0000 0009_1.5.1.zip` (Gen2 SDK v1.5.1) from the lab's Google Drive or Kinova.
2. Rename and place it in the `ada_feeding` directory you just cloned.

```
# From your Downloads folder or wherever you saved it:
mv ~/Downloads/"PS 0000 0009_1.5.1.zip" ./kinova_sdk.zip
```

### **4\. Build the Shared Docker Image**

Build the Articutool version of the project and tag it as `articutool-dev`. All researchers on this machine share this image.

```
docker build . --build-arg ROSINSTALL_FILE=articutool.https.rosinstall -t articutool-dev
```

The admin setup is now complete. The `articutool-dev` image is available for all users on the machine.

## **Part 2: New Researcher Workflow**
Welcome to the lab! Your workflow involves editing code in a local folder (`~/ada_ws`) and running it inside the pre-built `articutool-dev` container.

### **Step 1: First-Time Graphical Login**
You must do this one time to ensure your user is set up for GUI applications.
1. Log out of the lab machine.
2. At the Ubuntu login screen, log in as your new user (`<new_username>`).
3. Once the desktop loads, you're done. You can can open a terminal for the next steps.

### **Step 2: Download Your Local Code**
You only do this once to get your personal copy of the source code.
1. **Create your workspace:**
```
cd ~
mkdir -p ada_ws/src
cd ada_ws/src
```

2. **Clone the installer tool:** **Note:** We must use the `jjaime2/articutool-ros-install` branch to get the correct `.rosinstall` file.
```
# Clone the pr-rosinstalls repo and check out the articutool branch
git clone --branch jjaime2/articutool-ros-install https://github.com/personalrobotics/pr-rosinstalls.git
```

3. **Download all source code:**
```
# Install wstool if not already on the host machine
sudo apt install python3-wstool

# Download all repos specified in the .rosinstall file
wstool init
wstool merge pr-rosinstalls/articutool.httpsias.rosinstall
wstool up
```

You now have your `~/ada_ws/src` directory where you can edit files and manage git branches.

### **Step 3: Configure Local Network (One-Time)**
You must tell ROS 2 which network interface to use.
1. **Find your interface name:** Open a terminal and run `ifconfig`. Look for your main network (e.g., `wlx...` or `enp...`) and note its name.
2. **Edit the config file:**
```
vim ~/ada_ws/src/ada_feeding/cyclonedds.xml
```
3. **Add your interface:** Add a new `<NetworkInterface>` line with the name you just found.
```
<?xml version="1.0" encoding="utf-8"?>
<CycloneDDS
  xmlns="https://cdds.io/config"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd"
>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="lo" priority="default" multicast="true"/>
        <NetworkInterface name="<network_interface_name!!!>" priority="default" multicast="true"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>
```

### **Step 4: Run the Simulation Container**
This is the command you will run **every time** you want to start a simulation session.
1. **Allow GUI connections:** Open a terminal on the host machine and run:
```
xhost +local:
```

You must do this once per login session to allow the container to open GUI windows like RViz.

2. **Run the container:** This command mounts your local `~/ada_ws/src` folder into the container.
```
docker run -it --rm \
  --network=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/ada_ws/src:/home/ros/colcon_ws/src \
  -v /home/ros/colcon_ws/src/feeding_web_interface/feedingwebapp/node_modules \
  --add-host=babbage:192.168.4.50 \
  --add-host=nano:192.168.4.4 \
  articutool-dev
```
   * `--network=host`: For ROS 2 discovery and the web app.
   * `--privileged`: Gives the container access to USB hardware (like the Jaco arm and E-Stop)
   * `-e` and `-v ...X11...`: For X11 forwarding (GUI).
   * `-v ~/ada_ws/src...`: **(IMPORTANT)** Mounts your local code into the container.
   * `-v ...node_modules`: **(IMPORTANT)** Connects the pre-installed web app libraries.

This drops you into a bash prompt inside the container: `ros@...:~/colcon_ws$`

### **Step 5: Build and Run the Software (Inside Container)**
You are now inside the container.

1. **Compile your mounted code:**
```
colcon build --symlink-install
```

2. **Source your environment:**
```
source install/setup.bash
```

3. **Run the Start Script:** Use `start.py` to launch the software. The general command is:
```
python3 src/ada_feeding/start.py --sim <MODE> --dev --action <INDEX>
```

**Flag Explanations:**
* `--sim <MODE>`: Specify the robot mode.
   * `mock`: Use the mock robot for simulation.
   * `real`: User the real robot hardware (requires the `--privileged` container).
* `--dev`: Recommended. This launches RViz, does not require the e-stop button, and does not require `sudo` access for the web app.
* `--action <INDEX>`: An integer for the food acquisition action. These indices correspond to entries in the `.../config/acquisition_library.yaml` file.

You can access the web interface in a browser on your host machine at `http://localhost:3000` (when using `--dev` or `--sim mock`).

## **Manual Setup (Archival / For Docker Development)**

**Note:** These instructions are preserved for historical reference and for developers who are debugging the Dockerfile itself. For all normal development, demos, and research, please use the Docker setup above.

### Setup (Robot Software)

1. [Install ROS2 Humble](https://docs.ros.org/en/humble/Installation.html) (binary packages are recommended over building from source), [configure your environment](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html), and [create a workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html#).

   1. **NOTE**: In the remainder of the instructions, replace `~\colcon_ws` with the path to your workspace.

2. Configure [`pr-rosinstalls`](https://github.com/personalrobotics/pr-rosinstalls) in order to download all necessary repositories.

   ```
    cd ~
    git clone https://github.com/personalrobotics/pr-rosinstalls.git
   ```

3. Pull all the repositories, with the correct branch. Replace `<https/ssh>` with one of the options, depending on your authentication method.

   ```
    sudo apt install python3-wstool # if not already installed
    cd ~/colcon_ws/src
    wstool init
    wstool merge ~/pr-rosinstalls/ada-feeding.<https/ssh>.rosinstall
    wstool up
   ```

4. Configure [`rosdep`](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html):

   ```
    sudo apt install python3-rosdep # if not already installed
    sudo rosdep init # if this is the first time using rosdep
   ```

5. (Only for users **with** sudo access) Install [`rosdep`](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html) dependencies:

   ```
    rosdep update
    cd ~/colcon_ws
    rosdep install --from-paths src -y --ignore-src --as-root=pip:false
   ```

6. Install and fix pip/non-`rosdep` dependencies:

   ```
    cd ~/colcon_ws/src/ada_feeding
    python3 -m pip install -r requirements.txt
   ```

   - Upgrade `transforms3d`, since [the release on Ubuntu packages is outdated](https://github.com/matthew-brett/transforms3d/issues/65): `python3 -m pip install transforms3d -U`

   - Remove the duplicate matplotlib pip installation caused by installing scikit-spatial with pip (some accounts have required sudo access for this command, other have not. If you do not have sudo access and encounter this, contact a lab member who does):

     ```
       python3 -m pip uninstall matplotlib
     ```

   - [`pyrealsense2` is not released for ARM systems](https://github.com/IntelRealSense/librealsense/issues/6449#issuecomment-650784066), so ARM users will have to [build from source](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md#building-from-source). You may have to add the `-DPYTHON_EXECUTABLE=/usr/bin/python3` flag to the `cmake` command. When running `sudo make install`, pay close attention to which path `pyrealsense2` is installed to and add *that path* to the `PYTHONPATH` -- it should be `/use/local/lib` but may be `/usr/local/OFF`.

7. Install the JACO SDK (real robot only). All SDKs are listed [here](https://www.kinovarobotics.com/resources?r=79301&s); PRL currently uses the [Gen2 SDK v1.5.1](https://drive.google.com/file/d/1UEQAow0XLcVcPCeQfHK9ERBihOCclkJ9/view). Note that although the latest version of that SDK is for Ubuntu 16.04, it still works on Ubuntu 22.04 (only for x86 systems, not ARM system).

8. (Optional): We recommend using CycloneDDS as your ROS2 midleware, with a custom configuration file that enables multicast on loopback and prioritizes loopback. Install it with `sudo apt install ros-humble-rmw-cyclonedds-cpp`. Then, add the following lines to your `~/bashrc`: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; export CYCLONEDDS_URI=~/colcon_ws/src/ada_feeding/cyclonedds.xml`.

   - Note that you may have to change the name of the fallback interface that CycloneDDS uses if `lo` does not work for that transmission. To do so, run `ifconfig` and either use the name of your ethernet network or WiFi network, depending on how your computer is connected to the internet.
   - Note that you have to [rebuild your workspace from scratch after doing this](https://docs.ros.org/en/humble/How-To-Guides/Working-with-multiple-RMW-implementations.html#adding-rmw-implementations-to-your-workspace).

9. Build your workspace:

   ```
    cd ~/colcon_ws
    colcon build --symlink-install # if sim-only, add '--packages-skip ada_hardware'
   ```

### Setup (System Configuration)

Our system includes two computers, `lovelace` (Lenovo Legion 5i) and `nano` (Nvidia Jetson Nano), where the latter runs camera code and the former runs everything else. This repository includes some system-specific bash scripts to configure those computers. Specifically, move `start_nano.sh`, `start_nano.py`, `run_camera.sh`, and `run_nano_bridge.sh` to the home directory of `nano`. Copy `nano_bridge` into the ROS2 workspace on `nano` (e.g., `~/Workspace/camera_ros2`), and rebuild it. On `lovelace`, add a cronjob to run `configure_lovelace.sh` on reboot (e.g., `sudo crontab -e`, then add the line `@reboot <path/to/ada_feeding>/configure_lovelace.sh`).

Note that these scripts will likely need to be changed for your setup's computers; however, perhaps the commands in these scripts can provide you inspiration.

### Setup (Robot Hardware)

If you change the e-stop button, the e-stop button's adapter(s), and/or the device this code is being run on, you may have to change several device-specific parameters for the e-stop to work. Read the long comment above `amixer_configuration_name` in `ada_feeding/config/ada_watchdog.yaml` for detailed instructions on changing these parameters.

### Setup (Web App)

1. Install the Node Version Manager (nvm): https://github.com/nvm-sh/nvm?tab=readme-ov-file#install--update-script

2. Install and use NodeJS 21 (Note: if you have just installed nvm using the previous command, you will need to source your .bashrc or open a new terminal to run these commands):

   ```
    nvm install 21
    nvm use 21
   ```

3. (Only for users **with** sudo access; this should already be configured on PRL computers) Make Node available to all users, including root:

   ```
    sudo ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/node" "/usr/local/bin/node"
    sudo ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/npm" "/usr/local/bin/npm"
    sudo ln -s "$NVM_DIR/versions/node/$(nvm version)/bin/npx" "/usr/local/bin/npx"
   ```

4. (Only for users **with** sudo access; this should already be configured on PRL computers) Install `serve` and `pm2` globally. Root access is necessary for `serve` so it can access port 80.

   ```
    sudo npm install -g serve
    npm install -g pm2@latest
   ```

5. Install the web app dependencies. (Note: there will be some vulnerabilities in dependencies. This is okay, since access to the web app is shielded behind our secure router.)

   ```
    cd ~/colcon_ws/src/feeding_web_interface/feedingwebapp
    npm install --legacy-peer-deps
    npx playwright install
   ```

6. (Optional; this should already be configured on PRL computers) To access the web app on a device other than the one hosting it, enable the desired port for HTTP access: https://www.digitalocean.com/community/tutorials/ufw-essentials-common-firewall-rules-and-commands#allow-all-incoming-http-port-80

## Running the Software

We use the convenience script `start.py` to launch the software. This script has several command-line arguments, which can be seen by passing the `-h` flag when running the script.

### **Recommended** Option A: Run the Web App with the Real Robot

This option starts the web app and the real robot code, and can be used to test the entire system. This will by default start the web app on port `80`, and requires `sudo` access. The robot's HDMI connection must be plugged into your computer, and your computer should be connected to the robot's router with an ethernet cable.

**NOTE**: If not running on the production machine i.e., `lovelace`, it's recommended that you append the command line flag `--dev` to the start script. This will launch RVIZ, will not require the e-stop button to be plugged in, and will not require sudo access to launch the web app.

```
cd ~/colcon_ws
python3 src/ada_feeding/start.py
```

In a browser, access `127.0.0.1` (if on the same device serving the web app), or the IP address of the device hosting the web app (if on a different device connected to the same network, e.g. a cell phone connected to the LOVELACE_5g network). You should now be able to run the system! Note that upon startup, the watchdog is in a failing state until the e-stop is clicked exactly once, allowing the system to verify that it is connected and working.

To close, run `python3 src/ada_feeding/start.py -c`

### Option B: Running Web App With the Mock Robot

This option starts the web app, runs dummy nodes for perception, runs the **real** robot motion code, but runs a mock robot in MoveIt. This is useful for testing robot motion code in simulation before moving onto the real robot. This will start the web app on port `3000` and does not require `sudo` access.

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

## Contributing

1. Install `pre-commit`:
   1. `python3 -m pip install pre-commit`
   2. `cd ~/colcon_ws/src/ada_feeding`
   3. `pre-commit install`
2. Before opening a pull request:
   1. `cd ~/colcon_ws/src/ada_feeding`
   2. `pre-commit run --all-files`
   3. `pylint --recursive=y --rcfile=.pylintrc .`

## Troubleshooting

- **Something is not working, what do I do?**: All our code runs in `screen` sessions, so the first step is to attach to individual screen sessions to identify where the issue is. Run `screen -ls` to list all screens, run `screen -r <name>` to attach to a screen session, and `Ctrl-A d` to detach from the screen session. An introduction to `screen` can be found [here](https://astrobiomike.github.io/unix/screen-intro).
- **The watchdog is not recognizing my initial e-stop click**: Verify the e-stop is plugged in, and that any other processes accessing the microphone (e.g., Sound Settings) are closed. Run `alsamixer` to see if your user account has access to it. If you do not see sound levels in the terminal, try `sudo alsamixer`. If that works, give your user account permission to access sound: `sudo setfacl -m u:$USER:rw /dev/snd/*`
- **The watchdog is failing due to the F/T sensor**: First, check whether the force-torque sensor is publishing: `ros2 topic echo /wireless_ft/ftSensor3`. If not, the issue is in the `ft` screen (one potential issue is that the alias `ft-sensor` does not point to the right IP address for the force-torque sensor, in which case you should pass the IP address in using the `host` parameter). If so, check the timestamp of the published F/T messages compared to the current time. If it is off, the problem is that NTP got disabled on the force-torque sensor. You have to use `minicom` to re-enable NTP (see [here](https://github.com/personalrobotics/pr_docs/wiki/ADA) for PRL-specific instructions).
- **I get the error `cannot use destroyable because destruction was requested`**: Upgrade to the latest version of`rclpy`.
- **I get the error `bad option: --env-file=.env` when launching the WebRTC Signalling Server**: You are using an old version of Node; upgrade it to 21.
