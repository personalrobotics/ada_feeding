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
parser.add_argument("--sim", action="store_true", help="If set, run the code in sim")
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


async def terminate_code_in_screen(screen_name: str) -> bool:
    """
    Send Ctrl+C to the screen session.
    """
    proc = await asyncio.create_subprocess_shell(
        f"screen -S {screen_name} -X stuff $'\003'"
    )
    await proc.communicate()
    return proc.returncode == 0


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
        print(f"# Starting the ada_feeding demo in **{'sim' if args.sim else 'real'}**")
        print("# Prerequisites / Notes:")
        print("#     1. Be in the top-level of your colcon workspace")
        print(
            "#     2. Your workspace should be built (e.g., `colcon build --symlink-install`)"
        )
        if not args.sim:
            print(
                "#     3. The web app should be built (e.g., `npm run build` in "
                "`./src/feeding_web_interface/feedingwebapp`)."
            )
    else:
        print(
            f"# Terminating the ada_feeding demo in **{'sim' if args.sim else 'real'}**"
        )
    print(
        "################################################################################"
    )

    # Determine which screen sessions to start and what commands to run
    sudo_password = None
    if args.sim:
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
                    "src/ada_feeding/ada_feeding_perception/config/republisher.yaml",
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
    else:
        screen_sessions = {
            "web": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "sudo serve -s build -l 80",
            ],
            "webrtc": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "pm2 delete server",
                "pm2 start server.js",
            ],
            "camera": [
                "ssh nano './run_camera.sh'",
            ],
            "ft": [
                "ros2 run forque_sensor_hardware forque_sensor_hardware",
            ],
            "rosbridge": [
                "ros2 launch rosbridge_server rosbridge_websocket_launch.xml",
            ],
            "perception": [
                "ros2 launch ada_feeding_perception ada_feeding_perception.launch.py",
            ],
            "moveit": [
                "Xvfb :5 -screen 0 800x600x24 &",
                "export DISPLAY=:5",
                "ros2 launch ada_moveit demo.launch.py use_rviz:=false",
            ],
            "feeding": [
                "ros2 launch ada_feeding ada_feeding_launch.xml use_estop:=true run_web_bridge:=false",
            ],
            "browser": [
                "cd ./src/feeding_web_interface/feedingwebapp",
                "node start_robot_browser.js --port=80",
            ],
        }
    initial_commands = [
        f"cd {pwd}",
        "source install/setup.bash",
    ]
    for screen_name, commands in screen_sessions.items():
        screen_sessions[screen_name] = initial_commands + commands

    # Determine which screens are already running
    print("# Checking for existing screen sessions")
    terminated_screen = False
    existing_screens = await get_existing_screens()
    for screen_name in screen_sessions:
        if screen_name in existing_screens:
            print(f"#    Found session `{screen_name}`: ", end="")
            await asyncio.create_subprocess_shell(
                f"screen -S {screen_name} -X stuff $'\003'"
            )
            print("Sent SIGINT")
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
                print(f"#         `{command}`")
                await asyncio.create_subprocess_shell(
                    f"screen -S {screen_name} -X stuff '{command}\n'"
                )
                await asyncio.sleep(args.launch_wait_secs)
                if command.startswith("sudo"):
                    if sudo_password is None:
                        sudo_password = getpass.getpass(
                            prompt="#         Enter sudo password: "
                        )
                    await asyncio.create_subprocess_shell(
                        f"screen -S {screen_name} -X stuff '{sudo_password}\n'"
                    )
                    await asyncio.sleep(args.launch_wait_secs)
                elif command.startswith("ssh"):
                    ssh_password = getpass.getpass(
                        prompt="#         Enter ssh password: "
                    )
                    await asyncio.create_subprocess_shell(
                        f"screen -S {screen_name} -X stuff '{ssh_password}\n'"
                    )
                    await asyncio.sleep(args.launch_wait_secs)

        print(
            "################################################################################"
        )

        print("# Done! Next steps:")
        print(
            "#     1. Check individual screens to verify code is working as expected."
        )
        if not args.sim:
            print("#     2. Push the e-stop button to enable the robot.")
            print("#     3. Note that this script starts the app on port 80.")
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
    asyncio.run(main(args, pwd))

    # Return success
    sys.exit(0)
