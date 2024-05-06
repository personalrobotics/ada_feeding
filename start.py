#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module starts all the screen sessions to run the ada_feeding demo.
"""

import asyncio
import argparse
import getpass
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sim",
    default="real",
    help=(
        "`real`, `mock`, or `dummy` (default `real`). `real` executes all commands for running "
        "the feeding code on the real robot, `mock` uses dummy perception and real motion code "
        "with a simulated robot, and `dummy` uses dummy perception and motion code. All three "
        "launch the web app."
    ),
)
parser.add_argument(
    "-t",
    "--termination_wait_secs",
    default=5,
    help="How long (secs) to wait for the code within screens to terminate (default, 5)",
)
parser.add_argument(
    "-l",
    "--launch_wait_secs",
    default=0.1,
    help=(
        "How long (secs) to wait between running code in screens. Not that "
        "too low of a value can result in commands getting rearranged. (default, 0.1)"
    ),
)
parser.add_argument(
    "-c",
    "--close",
    action="store_true",
    help="If set, only terminate the code in the screens.",
)
parser.add_argument(
    "--dev",
    action="store_true",
    help=(
        "If set, do not use the e-stop in `feeding`, and show RVIZ in `moveit`. "
        "Also launch the web app on port 3000. Only applies to `real`."
    ),
)
parser.add_argument(
    "--real_domain_id",
    default=42,
    type=int,
    help=(
        "If sim is set to real, export this ROS_DOMAIN_ID before running code in "
        "every screen session. (default: 42)"
    ),
)


async def get_existing_screens():
    """
    Get a list of active screen sessions.

    Adapted from
    https://serverfault.com/questions/886405/create-screen-session-in-background-only-if-it-doesnt-already-exist
    """
    proc = await asyncio.create_subprocess_shell(
        "screen -ls", stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    existing_screens = [
        line.split(".")[1].split("\t")[0].rstrip()
        for line in stdout.decode("utf-8").splitlines()
        if line.startswith("\t")
    ]
    return existing_screens


async def execute_command(
    screen_name: str, command: str, indent: int = 8
) -> None:
    """
    Execute a command in a screen.
    """
    global sudo_password # pylint: disable=global-statement
    indentation = " " * indent
    printable_command = command.replace("\003", "SIGINT")
    print(f"# {indentation}`{printable_command}`")
    await asyncio.create_subprocess_shell(
        f"screen -S {screen_name} -X stuff '{command}\n'"
    )
    await asyncio.sleep(args.launch_wait_secs)
    if command.startswith("sudo"):
        if sudo_password is None:
            sudo_password = getpass.getpass(
                prompt=f"# {indentation}Enter sudo password: "
            )
        await asyncio.create_subprocess_shell(
            f"screen -S {screen_name} -X stuff '{sudo_password}\n'"
        )
        await asyncio.sleep(args.launch_wait_secs)
    elif command.startswith("ssh"):
        ssh_password = getpass.getpass(prompt=f"# {indentation}Enter ssh password: ")
        await asyncio.create_subprocess_shell(
            f"screen -S {screen_name} -X stuff '{ssh_password}\n'"
        )
        await asyncio.sleep(args.launch_wait_secs)


async def main(args: argparse.Namespace, pwd: str) -> None:
    """
    Start the ada_feeding demo.

    Args:
        args: The command-line arguments.
        pwd: The absolute path to the current directory.
    """

    # pylint: disable=too-many-branches, too-many-statements
    # This is meant to be a flexible function, hence the many branches and statements.
    # pylint: disable=redefined-outer-name
    # That is okay in this case.

    print(
        "################################################################################"
    )
    if not args.close:
        print(f"# Starting the ada_feeding demo in **{args.sim}**")
        print("# Prerequisites / Notes:")
        print("#     1. Be in the top-level of your colcon workspace")
        print(
            "#     2. Your workspace should be built (e.g., `colcon build --symlink-install`)"
        )
        if args.sim == "real":
            print(
                "#     3. The web app should be built (e.g., `npm run build` in "
                "`./src/feeding_web_interface/feedingwebapp`)."
            )
    else:
        print(f"# Terminating the ada_feeding demo in **{ args.sim}**")
    print(
        "################################################################################"
    )

    # Determine which screen sessions to start and what commands to run
    if args.sim == "mock":
        screen_sessions = {
            "web": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "npm run start",
            ],
            "webrtc": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "node --env-file=.env server.js",
            ],
            "ft": [
                "ros2 run ada_feeding dummy_ft_sensor.py",
            ],
            "camera": [
                (
                    "ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml "
                    "run_motion:=false run_web_bridge:=false "
                    "run_food_detection:=false run_face_detection:=false "
                    "run_food_on_fork_detection:=false run_table_detection:=false "
                ),
            ],
            "perception": [
                (
                    "ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml "
                    "run_motion:=false run_web_bridge:=false run_real_sense:=false"
                ),
            ],
            "republisher": [
                (
                    "ros2 run ada_feeding_perception republisher --ros-args --params-file "
                    "src/ada_feeding/ada_feeding_perception/config/republisher.yaml"
                ),
            ],
            "rosbridge": [
                "ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
            ],
            "feeding": [
                "ros2 launch ada_feeding ada_feeding_launch.xml use_estop:=false"
            ],
            "moveit": ["ros2 launch ada_moveit demo.launch.py sim:=mock"],
            "browser": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "node start_robot_browser.js",
            ],
        }
        close_commands = {}
    elif args.sim == "dummy":
        screen_sessions = {
            "web": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "npm run start",
            ],
            "webrtc": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "node --env-file=.env server.js",
            ],
            "ft": [
                "ros2 run ada_feeding dummy_ft_sensor.py",
            ],
            "perception": [
                (
                    "ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml "
                    "run_motion:=false run_web_bridge:=false"
                ),
            ],
            "republisher": [
                (
                    "ros2 run ada_feeding_perception republisher --ros-args --params-file "
                    "src/ada_feeding/ada_feeding_perception/config/republisher.yaml"
                ),
            ],
            "rosbridge": [
                "ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
            ],
            "feeding": [
                (
                    "ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml "
                    "run_web_bridge:=false run_food_detection:=false run_face_detection:=false "
                    "run_food_on_fork_detection:=false run_table_detection:=false "
                    "run_real_sense:=false"
                ),
            ],
            "browser": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "node start_robot_browser.js",
            ],
        }
        close_commands = {}
    else:
        screen_sessions = {
            "web": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "npm run start" if args.dev else "sudo npx serve -s build -l 80",
            ],
            "webrtc": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "pm2 start server.js",
                "pm2 log server",
            ],
            "camera": [
                "ssh nano './run_camera.sh'",
            ],
            "ft": [
                "ros2 run forque_sensor_hardware forque_sensor_hardware --ros-args -p host:=ft-sensor-2",
            ],
            "rosbridge": [
                "ros2 launch rosbridge_server rosbridge_websocket_launch.xml",
            ],
            "perception": [
                "ros2 launch ada_feeding_perception ada_feeding_perception.launch.py",
            ],
            "moveit": [
                "Xvfb :5 -screen 0 800x600x24 &" if not args.dev else "",
                "export DISPLAY=:5" if not args.dev else "",
                f"ros2 launch ada_moveit demo.launch.py use_rviz:={'true' if args.dev else 'false'}",
            ],
            "feeding": [
                (
                    "ros2 launch ada_feeding ada_feeding_launch.xml "
                    f"use_estop:={'false' if args.dev else 'true'} run_web_bridge:=false"
                ),
            ],
            "browser": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "node start_robot_browser.js" + "" if args.dev else " --port=80",
            ],
        }
        close_commands = {
            "webrtc": [
                "pm2 delete server",
            ]
        }
    initial_close_commands = ["\003"]
    initial_start_commands = [
        f"cd {pwd}",
        "source install/setup.bash",
    ]
    if args.sim == "real":
        initial_start_commands.append(f"export ROS_DOMAIN_ID={args.real_domain_id}")
    for screen_name, commands in screen_sessions.items():
        screen_sessions[screen_name] = initial_start_commands + commands
        if screen_name not in close_commands:
            close_commands[screen_name] = []
        close_commands[screen_name] = (
            initial_close_commands + close_commands[screen_name]
        )

    # Determine which screens are already running
    print("# Checking for existing screen sessions")
    terminated_screen = False
    existing_screens = await get_existing_screens()
    for screen_name in screen_sessions:
        if screen_name in existing_screens:
            print(f"#    Found session `{screen_name}`: ")
            for command in close_commands[screen_name]:
                await execute_command(screen_name, command)
            terminated_screen = True
        elif not args.close:
            print(f"#    Creating session `{screen_name}`")
            await asyncio.create_subprocess_shell(f"screen -dmS {screen_name}")
            await asyncio.sleep(args.launch_wait_secs)

    print(
        "################################################################################"
    )

    if not args.close:
        # Sleep for a bit to allow the screens to terminate
        if terminated_screen:
            print(f"# Waiting {args.termination_wait_secs} secs for code to terminate")
            await asyncio.sleep(args.termination_wait_secs)
            print(
                "################################################################################"
            )

        # Start the screen sessions
        print("# Starting robot feeding code")
        for screen_name, commands in screen_sessions.items():
            print(f"#     `{screen_name}`")
            for command in commands:
                await execute_command(screen_name, command)
        print(
            "################################################################################"
        )

        print("# Done! Next steps:")
        print(
            "#     1. Check individual screens to verify code is working as expected."
        )
        if args.sim == "real":
            print("#     2. Push the e-stop button to enable the robot.")
            print(
                f"#     3. Note that this script starts the app on port {3000 if args.dev else 80}."
            )
        else:
            print("#     2. Note that this script starts the app on port 3000.")


def check_pwd_is_colcon_workspace() -> str:
    """
    Check that the script is being run from the top-level colcon workspace.
    Return the absolute path to the current directory.
    """
    # The below are a smattering of directories and files that should exist in the
    # top-level colcon workspace.
    dirs_to_check = ["src", "build", "install", "log"]
    files_to_check = [
        "src/feeding_web_interface/feedingwebapp/server.js",
        "src/feeding_web_interface/feedingwebapp/.env",
        "src/feeding_web_interface/feedingwebapp/start_robot_browser.js",
        "src/ada_feeding/ada_feeding_perception/config/republisher.yaml",
    ]
    for dir_to_check in dirs_to_check:
        if not os.path.isdir(dir_to_check):
            print(
                f"ERROR: This script must be run from the top-level colcon workspace. Could not find `{dir_to_check}`",
                file=sys.stderr,
            )
            sys.exit(1)
    for file_to_check in files_to_check:
        if not os.path.isfile(file_to_check):
            print(
                f"ERROR: This script must be run from the top-level colcon workspace. Could not find `{file_to_check}`",
                file=sys.stderr,
            )
            sys.exit(1)

    # Return the absolute path to the current directory
    return os.path.abspath(".")


if __name__ == "__main__":
    # Get the arguments
    args = parser.parse_args()
    # Check args
    if args.sim not in ["real", "mock", "dummy"]:
        raise ValueError(
            f"Unknown sim value {args.sim}. Must be one of ['real', 'mock', 'dummy']."
        )

    # Ensure the script is not being run as sudo. Sudo has a different screen
    # server and may have different versions of libraries installed.
    if os.geteuid() == 0:
        print(
            "ERROR: This script should not be run as sudo. Run as a regular user.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check that the script is being run from the top-level colcon workspace
    pwd = check_pwd_is_colcon_workspace()

    # Run the main function
    sudo_password = None # pylint: disable=invalid-name
    asyncio.run(main(args, pwd))

    # Return success
    sys.exit(0)
