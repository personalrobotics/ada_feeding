# 1. Start from the official ROS 2 Humble base image
FROM ros:humble-ros-base

# Add a default rosinstall file argument
ARG ROSINSTALL_FILE=articutool.https.rosinstall

# Set non-interactive mode for apt to prevent it from hanging
ENV DEBIAN_FRONTEND=noninteractive

# Install core tools
RUN apt-get update && apt-get install -y \
    git \
    python3-wstool \
    python3-rosdep \
    unzip \
    curl \
    lsb-release \
    gnupg2 \
    python3 \
    python3-pip \
    python3-venv \
    net-tools \
    iputils-ping \
    vim \
    git \
    build-essential \
    cmake \
    python3-dev \
    libboost-all-dev \
    libeigen3-dev \
    portaudio19-dev \
    tzdata \
    usbutils \
    sudo \
    screen \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-joint-state-publisher \
    ros-humble-joint-trajectory-controller \
    ros-humble-diff-drive-controller \
    ros-humble-xacro \
    ros-humble-joint-state-publisher-gui \
    ros-humble-dynamixel-sdk \
    ros-humble-dynamixel-workbench-toolbox \
    ros-humble-pinocchio \
    ros-humble-tf-transformations \
    ros-humble-imu-tools \
    ros-humble-rmw-cyclonedds-cpp

RUN useradd -m -s /bin/bash ros && \
    echo "ros:ros" | chpasswd && \
    adduser ros sudo

# Give the new 'ros' user passwordless sudo
RUN echo "ros ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install ssh client tools
USER root
RUN apt-get update && apt-get install -y ssh-client
USER ros

# --- Pre-trust SSH hosts ---
# We now scan for both hostname AND IP to create a comprehensive known_hosts file.
RUN mkdir -p $HOME/.ssh && \
    touch $HOME/.ssh/known_hosts && \
    chmod 700 $HOME/.ssh && \
    ssh-keyscan -H babbage,192.168.4.50 >> $HOME/.ssh/known_hosts && \
    ssh-keyscan -H nano,192.168.4.4 >> $HOME/.ssh/known_hosts && \
    chmod 600 $HOME/.ssh/known_hosts

# Set the user for the rest of the build
USER ros
ENV HOME=/home/ros
WORKDIR $HOME

# 4. Configure pr-rosinstalls
RUN git clone --branch jjaime2/articutool-ros-install https://github.com/personalrobotics/pr-rosinstalls.git $HOME/pr-rosinstalls
RUN mkdir -p $HOME/colcon_ws/src

# 5. WORKDIR
WORKDIR $HOME/colcon_ws/src

# 6. wstool init
RUN wstool init

# 7. wstool merge
RUN wstool merge $HOME/pr-rosinstalls/$ROSINSTALL_FILE

# 8. wstool up
RUN wstool up

# 9. WORKDIR
WORKDIR $HOME/colcon_ws

# 10. rosdep update
RUN rosdep update

# 11. rosdep install
RUN rosdep install --from-paths src -y --ignore-src --as-root=pip:false

# 12. COPY Kinova SDK
COPY kinova_sdk.zip /tmp/

# 13. Install Kinova SDK
RUN cd /tmp && \
    unzip kinova_sdk.zip && \
    sudo dpkg -i "Ubuntu/16_04/64 bits/KinovaAPI-6.1.0-amd64.deb" && \
    sudo rm -rf /tmp/*

# 14. Setup NVM
ENV NVM_DIR=$HOME/.nvm
ENV NODE_VERSION=21
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
ENV PATH="$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH"

# 15. Install global npm packages
RUN . "$NVM_DIR/nvm.sh" && npm install -g serve pm2@latest

# 16. Install pip requirements
# This COPY command makes this step cache-aware.
# If requirements.txt changes, this step and all following steps will re-run.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install transforms3d -U
RUN pip uninstall -y matplotlib

# 17. Install web app dependencies
WORKDIR $HOME/colcon_ws/src/feeding_web_interface/feedingwebapp
RUN . "$NVM_DIR/nvm.sh" && npm install --legacy-peer-deps
RUN . "$NVM_DIR/nvm.sh" && npx playwright install

# 18. Build workspace
WORKDIR $HOME/colcon_ws
RUN . /opt/ros/humble/setup.sh && \
    . "$NVM_DIR/nvm.sh" && \
    colcon build --symlink-install --packages-skip ada_hardware

# Force 'screen' to always use bash for interactive shells
RUN echo "shell /bin/bash" > /home/ros/.screenrc

RUN mkdir -p $HOME/colcon_ws/install/ada_feeding_action_select/share/ada_feeding_action_select/data/checkpoint/adapter
# 19. Configure Environment
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=$HOME/colcon_ws/src/ada_feeding/cyclonedds.xml

# 20. Final Configuration
# We copy the entrypoint script and set permissions
USER root
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
USER ros
ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["bash"]
