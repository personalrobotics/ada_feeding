#!/bin/bash
set -e

source "/opt/ros/$ROS_DISTRO/setup.bash"
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.01
unset ROS_HOSTNAME
exec "$@"
