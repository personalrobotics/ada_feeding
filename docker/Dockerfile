FROM ros:noetic
RUN apt update && \
apt-get install -y -qq iputils-ping \
git \
libblas-dev \
libboost-all-dev \
liblapack-dev \
libeigen3-dev \
libmicrohttpd-dev \
libopencv-dev \
python3-pip \
python3-scipy \
python3-wstool \
python-is-python3 \
ros-noetic-catkin \
ros-noetic-control-toolbox \
ros-noetic-controller-manager \
ros-noetic-cv-bridge \
ros-noetic-diagnostic-updater \
ros-noetic-force-torque-sensor-controller \
ros-noetic-joint-trajectory-controller \
ros-noetic-geometry \
ros-noetic-geometry-msgs \
ros-noetic-image-geometry \
ros-noetic-image-transport \
ros-noetic-message-generation \
ros-noetic-message-runtime \
ros-noetic-octomap-ros \
ros-noetic-ompl \
ros-noetic-robot-state-publisher \
ros-noetic-rosbash \
ros-noetic-roscpp \
ros-noetic-roslaunch \
ros-noetic-rospy \
ros-noetic-rviz \
ros-noetic-std-msgs \
ros-noetic-srdfdom \
ros-noetic-tf2-ros \
ros-noetic-transmission-interface \
software-properties-common \
tmux \
vim
RUN ["pip", "install", "-U", "git+https://github.com/catkin/catkin_tools.git#egg=catkin_tools"]
RUN ["pip", "install", "pybind11[global]"]
RUN apt-add-repository ppa:dartsim/ppa
RUN apt-get install -y -qq iputils-ping \
libdart6-all-dev
COPY ./entrypoint.sh /
RUN ["chmod", "+x", "/entrypoint.sh"]
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
