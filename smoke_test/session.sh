#!/bin/bash
# Copyright (c) 2026, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

# Runs inside the container (see run.sh): starts dummy mode, checks that every
# process the meal needs is up, drives one meal through the web app, and
# collects configuration, logs, and events into the artifacts directory.
#
#   session.sh <artifacts_dir>

out="$1"
here="$(cd "$(dirname "$0")" && pwd)"
ws=/home/ros/colcon_ws
source /opt/ros/humble/setup.bash
source "$ws/install/setup.bash"
source "$NVM_DIR/nvm.sh"
set -uo pipefail

mkdir -p "$out/logs" "$out/config"
cd "$ws"

collect() {
    env | grep -E '^(ROS_|RMW_|CYCLONEDDS_)' | sort > "$out/config/environment.txt"
    echo "python3 src/ada_feeding/start.py --sim dummy" > "$out/config/start_command.txt"
    cp src/ada_feeding/ada.repos "$out/config/ada.repos"
    vcs export --exact src > "$out/config/sources_exact.repos" 2> /dev/null
    {
        python3 --version
        echo "node $(node --version)"
        pip freeze
    } > "$out/config/python_node_versions.txt"
    dpkg-query -W 'ros-humble-*' > "$out/config/ros_apt_packages.txt"
    cp -r ~/.ros/log "$out/logs/ros"
}

fail() {
    echo "FAIL: $*"
    collect
    exit 1
}

# Wait until `ros2 <kind> list` includes every given name.
wait_for_ros() {
    local kind=$1 deadline=$((SECONDS + 90)) have missing
    shift
    while :; do
        have=$(ros2 "$kind" list 2> /dev/null)
        missing=()
        for name in "$@"; do
            grep -qx -- "$name" <<< "$have" || missing+=("$name")
        done
        [ ${#missing[@]} -eq 0 ] && return 0
        [ $SECONDS -ge $deadline ] && fail "these ROS ${kind}s never started: ${missing[*]}"
        sleep 2
    done
}

# Wait until a URL answers with any HTTP status.
wait_for_http() {
    local url=$1 deadline=$((SECONDS + 180))
    until [ "$(curl -s -o /dev/null -w '%{http_code}' "$url")" != 000 ]; do
        [ $SECONDS -ge $deadline ] && fail "nothing is serving $url"
        sleep 2
    done
}

# Log every screen session that start.py creates.
printf 'deflog on\nlogfile %s/logs/screen-%%S.log\nlogfile flush 1\n' "$out" >> ~/.screenrc
if ! python3 src/ada_feeding/start.py --sim dummy > "$out/logs/start.log" 2>&1; then
    fail "start.py exited with an error (see logs/start.log)"
fi

echo "Waiting for dummy mode to come up"
wait_for_ros node /rosbridge_websocket /republisher /DummyRealSense /FaceDetection \
    /FoodOnForkDetection /TableDetection
wait_for_ros action /AcquireFood /MoveAbovePlate /MoveFromMouth /MoveFromMouthToAbovePlate \
    /MoveFromMouthToRestingPosition /MoveFromMouthToStagingConfiguration /MoveToMouth \
    /MoveToRestingPosition /MoveToStagingConfiguration /MoveToStowLocation /SegmentFromPoint
if ! timeout 30 ros2 topic echo --once /local/camera/color/image_raw/compressed --no-arr > /dev/null 2>&1; then
    fail "no images on /local/camera/color/image_raw/compressed, which the web app shows"
fi
wait_for_http http://localhost:3000
wait_for_http http://localhost:5000

# Record every action's goal status (accepted, canceled, succeeded, ...).
ros2 bag record -o "$out/rosbag" --include-hidden-topics -e '.*/_action/status' \
    > "$out/logs/rosbag_record.log" 2>&1 &
bag=$!

echo "Driving one meal through the web app"
NODE_PATH="$ws/src/ada_feeding/feeding_web_interface/feedingwebapp/node_modules" node "$here/meal.js" "$out"
status=$?

kill -INT $bag
wait $bag
ros2 bag info "$out/rosbag" > "$out/rosbag_info.txt" 2>&1
[ $status -eq 0 ] || fail "a meal step failed (see result.json)"
if ! grep -q '\[MoveAbovePlate\]: Goal canceled' ~/.ros/log/*/launch.log; then
    fail "the pause never canceled the MoveAbovePlate goal"
fi
collect
echo "PASS"
