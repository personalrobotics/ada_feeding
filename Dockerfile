# 1. Start from the official ROS 2 Humble base image
FROM ros:humble-ros-base@sha256:dfc94c85d8e01f230951b2a85ca5576e08473081c3da5fbbe8f3ab844703b9ba

# Set non-interactive mode for apt to prevent it from hanging
ENV DEBIAN_FRONTEND=noninteractive

# Install core tools
RUN apt-get update && apt-get install -y \
    git \
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
# We "pre-bake" the known public keys for the lab machines to avoid
# a network dependency during the build.
RUN mkdir -p $HOME/.ssh && \
    touch $HOME/.ssh/known_hosts && \
    chmod 700 $HOME/.ssh && \
    \
    # Keys for babbage,192.168.4.50
    echo "|1|49Ea8J/IKuDiiPlq0KhUdb3xYlg=|wK+sSomPzi+Had5r5MxHXoCODdw= ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDwLNAnsiJA6IaQQ2owzltW1DbVKlCbXn48UupSPx05qyaNJtc/zczW5Zsxp8JJhRZdzlnOkjDwinoRXRHLkbnmilyLjkS3GjMZIIonSRiTJEF1Fhb291JO83o0JQTPajh7OZBKQvTO/1/oLEl1KcggDueBPMyjKnG6LWDcLKqQ1diTKCKGoQpEmG01A/rWVXI+5guyMGYyXGj+pkY/C+bYBWVVQ1RdLjkQxbe8z2BQ1H3AWA2i8Eqd/lGfYQ+1YoApqh5zxJ0hadxK/zlyZb72SwzU/yTqTMdt87A9nbqU6lo0oja6KvNZLvBqYf3Cb/Eja+7mKa3jUtxXIleN/HeVW/fj+PcXBs66+lrhO58DVV4YwIf1/T30cT0SYFNQq+XrMf9ETWUYFpAM7MfPE/3Hvh1UyMUL73QfgQ8DuGQIusqxciSGqd+RAK5KX06+bsHKVYQ4+X/m7etSgFOZ68gDBOFX9NDj5SwUc4xCVqkg9O+U5RXLnsM6hPA983nWofc=" >> $HOME/.ssh/known_hosts && \
    echo "|1|YLqSxQZzlSg4MzjgVxDiSKn6kdU=|pT73D0e781iwFr2QvbDUWs8qGo8= ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDwLNAnsiJA6IaQQ2owzltW1DbVKlCbXn48UupSPx05qyaNJtc/zczW5Zsxp8JJhRZdzlnOkjDwinoRXRHLkbnmilyLjkS3GjMZIIonSRiTJEF1Fhb291JO83o0JQTPajh7OZBKQvTO/1/oLEl1KcggDueBPMyjKnG6LWDcLKqQ1diTKCKGoQpEmG01A/rWVXI+5guyMGYyXGj+pkY/C+bYBWVVQ1RdLjkQxbe8z2BQ1H3AWA2i8Eqd/lGfYQ+1YoApqh5zxJ0hadxK/zlyZb72SwzU/yTqTMdt87A9nbqU6lo0oja6KvNZLvBqYf3Cb/Eja+7mKa3jUtxXIleN/HeVW/fj+PcXBs66+lrhO58DVV4YwIf1/T30cT0SYFNQq+XrMf9ETWUYFpAM7MfPE/3Hvh1UyMUL73QfgQ8DuGQIusqxciSGqd+RAK5KX06+bsHKVYQ4+X/m7etSgFOZ68gDBOFX9NDj5SwUc4xCVqkg9O+U5RXLnsM6hPA983nWofc=" >> $HOME/.ssh/known_hosts && \
    echo "|1|zEtEQrjMQT3B6kPHgsb/6EZjHZY=|eCiGTvNte3FLFv0wDbxO8V+qCo4= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFiunkK0t3pBUYfUun1pNmcdss7ZNWP4WLIKpL3CQv2LdylcM6srKeVkiXS7GMvnpgacmXmuNDJhgltj1VjlpRs=" >> $HOME/.ssh/known_hosts && \
    echo "|1|gon+l4h21AEk1xTMb/r51fUERjY=|4LeKTthihgoFxV7zxy19Cuwfi2I= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFiunkK0t3pBUYfUun1pNmcdss7ZNWP4WLIKpL3CQv2LdylcM6srKeVkiXS7GMvnpgacmXmuNDJhgltj1VjlpRs=" >> $HOME/.ssh/known_hosts && \
    echo "|1|LM0LKILGN+VlrcQzEXM9ogYX3tg=|N8qmNqTaZf6f4zbEMH/VwPY9sFQ= ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF5kaTGWJITm9CM75x5++mNQyBM1Hxiabs8ucTanMBKa" >> $HOME/.ssh/known_hosts && \
    echo "|1|PEAm29WqwueLOvWxnt/CCk1HyJU=|Hec3gUZ7WVqiJxobmdcPaMeihrU= ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIF5kaTGWJITm9CM75x5++mNQyBM1Hxiabs8ucTanMBKa" >> $HOME/.ssh/known_hosts && \
    \
    # Keys for nano,192.168.4.4
    echo "|1|2Y0YXbnrNBg5E2fAPvSA7CPbdBM=|XqLcG4rQNf22WSiKhAR0siC3Zd0= ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMs4PtLKyG0UQY2GygkmQtTZGjqB4uvmZIRO0v9Ah34g" >> $HOME/.ssh/known_hosts && \
    echo "|1|bMR2xxJknX2KHFckxwyLc2IQdQE=|wdKgKePqGQ3BZ8sdsq0xdyZNFNg= ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMs4PtLKyG0UQY2GygkmQtTZGjqB4uvmZIRO0v9Ah34g" >> $HOME/.ssh/known_hosts && \
    echo "|1|q21hHbzafG7fraNHiEsb+uPyFyc=|nv3h7kDtfjzbK/rkpLIAo72vqHo= ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDeN41SxRLDs6QnbyiSYTBIiCr6D9ihcfXdHeA7sMchCMiCUnTM6yVj2xMdIdVIk9oh+PNh5HN/Z4S98rJHxgRIADcjPQ4M19UK2O6fMVdn8g1GvdrMTsMAo+SAmY8fMjj9XWNR+JR1mru9+Sf1EgJ/yzzj2Z+C+8brNJFYky89JblqDnYzeD/XVvzc9E78b8Qnro55+0A/0ms0QAulbACwlBBUNE/o+G7+caV024Q4gx7VjQALwrc1qxWl1/Ekh7c64yHfO7caC3cpHuvICn5COBKUc0TIEL2wmXiNEtP2quKFr/1ZreIWHjLDCRBPsNZRwTli9N+GeWP+ZP9iab9n" >> $HOME/.ssh/known_hosts && \
    echo "|1|C+TkorkKWD5Mk5+ZHAcpE6XFafI=|KxsqH0jhdRmXsPphRKEF9DlwI6o= ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDeN41SxRLDs6QnbyiSYTBIiCr6D9ihcfXdHeA7sMchCMiCUnTM6yVj2xMdIdVIk9oh+PNh5HN/Z4S98rJHxgRIADcjPQ4M19UK2O6fMVdn8g1GvdrMTsMAo+SAmY8fMjj9XWNR+JR1mru9+Sf1EgJ/yzzj2Z+C+8brNJFYky89JblqDnYzeD/XVvzc9E78b8Qnro55+0A/0ms0QAulbACwlBBUNE/o+G7+caV024Q4gx7VjQALwrc1qxWl1/Ekh7c64yHfO7caC3cpHuvICn5COBKUc0TIEL2wmXiNEtP2quKFr/1ZreIWHjLDCRBPsNZRwTli9N+GeWP+ZP9iab9n" >> $HOME/.ssh/known_hosts && \
    echo "|1|hzTc2C/DNzv5lhwhPvOkqJ8D/k8=|C2EmuGI97pgtXat1AmzxBM1WB+8= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBHG0KGLLjvhCsPWAowtNapo+H49ORNvksA5/T4jPEvkWB77SPHnJAcTlGm0p3gi2nnFsRSpnpRDja9+dM3WYg+Y=" >> $HOME/.ssh/known_hosts && \
    echo "|1|Hf4F9psUIaNVLIdK9x6c3BRi9zM=|h9ekiLF6+E9Ygo7Z6Wz/jbNHDBY= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBHG0KGLLjvhCsPWAowtNapo+H49ORNvksA5/T4jPEvkWB77SPHnJAcTlGm0p3gi2nnFsRSpnpRDja9+dM3WYg+Y=" >> $HOME/.ssh/known_hosts && \
    \
    # Set final permissions
    chmod 600 $HOME/.ssh/known_hosts

# Switch back to the ros user for the rest of the build
USER ros
ENV HOME=/home/ros
WORKDIR $HOME

# 4. Get the source (this checkout and every repository in ada.repos)
RUN mkdir -p $HOME/colcon_ws/src
WORKDIR $HOME/colcon_ws
COPY --chown=ros:ros ada.repos /tmp/ada.repos
RUN vcs import --input /tmp/ada.repos src
COPY --chown=ros:ros . src/ada_feeding

# 10. rosdep update, with rosdep's key database pinned to a rosdistro commit.
#    Upstream later removed keys this workspace uses (ros/rosdistro#51719, #51720).
ARG ROSDISTRO_COMMIT=72e6c7ae6a347bed3ab2230657802d26cb9e5bd1
ENV ROSDISTRO_INDEX_URL=https://raw.githubusercontent.com/ros/rosdistro/${ROSDISTRO_COMMIT}/index-v4.yaml
RUN sudo sed -i "s#ros/rosdistro/master/#ros/rosdistro/${ROSDISTRO_COMMIT}/#" \
      /etc/ros/rosdep/sources.list.d/20-default.list && \
    rosdep update

# 11. rosdep install, refreshing apt's lists first since cached ones go stale
RUN sudo apt-get update && \
    rosdep install --from-paths src -y --ignore-src --as-root=pip:false

# 12. Install the Kinova SDK (real robot only, x86 only, needs kinova_sdk.zip)
ARG WITH_KINOVA_SDK=false
RUN if [ "$WITH_KINOVA_SDK" = true ]; then \
      cd /tmp && \
      unzip $HOME/colcon_ws/src/ada_feeding/kinova_sdk.zip && \
      sudo dpkg -i "Ubuntu/16_04/64 bits/KinovaAPI-6.1.0-amd64.deb" && \
      sudo rm -rf /tmp/*; \
    fi

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

# 18. Build workspace (skipping ada_hardware without the Kinova SDK)
WORKDIR $HOME/colcon_ws
RUN . /opt/ros/humble/setup.sh && \
    . "$NVM_DIR/nvm.sh" && \
    colcon build --symlink-install \
      $([ "$WITH_KINOVA_SDK" = true ] || echo --packages-skip ada_hardware)

# Force 'screen' to always use bash for interactive shells
RUN echo "shell /bin/bash" > /home/ros/.screenrc

RUN mkdir -p $HOME/colcon_ws/install/ada_feeding_action_select/share/ada_feeding_action_select/data/checkpoint/adapter
# 19. Configure Environment
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=$HOME/colcon_ws/src/ada_feeding/cyclonedds.xml
ENV ROS_DOMAIN_ID=42

# 20. Final Configuration
# We copy the entrypoint script and set permissions
USER root
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
USER ros
ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["bash"]
