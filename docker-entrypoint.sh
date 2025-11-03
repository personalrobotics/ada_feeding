#!/bin/bash
set -e

# Source ROS 2
. /opt/ros/humble/setup.bash

# Source your workspace
. /home/ros/colcon_ws/install/setup.bash

# Source NVM
export NVM_DIR="/root/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Execute the command passed to the container (e.g., the CMD)
exec "$@"
