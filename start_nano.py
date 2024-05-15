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


async def execute_command(screen_name: str, command: str, indent: int = 8) -> None:
    """
    Execute a command in a screen.
    """
    global sudo_password  # pylint: disable=global-statement
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
        print("# Running nano's camera in a screen session")
        print("# Prerequisites / Notes:")
        print(
            "#     1. The home directory should contain the run_camera.sh script to "
            "start the camera."
        )
    else:
        print("# Terminating nano's camera in a screen session")
    print(
        "################################################################################"
    )

    # Determine which screen sessions to start and what commands to run
    screen_sessions = {
        "camera": [
            "~/run_camera.sh",
        ],
    }
    close_commands = {}
    attach_to_screen_name = "camera"
    initial_close_commands = ["\003"]  # Ctrl+c termination
    initial_start_commands = [f"cd {pwd}"]
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
        print("# Starting camera code")
        for screen_name, commands in screen_sessions.items():
            print(f"#     `{screen_name}`")
            for command in commands:
                await execute_command(screen_name, command)
        print(
            "################################################################################"
        )

        print(f"# Done! Attaching to the {attach_to_screen_name} screen session")
        await asyncio.create_subprocess_shell(f"screen -rd {attach_to_screen_name}")


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

    # Get the path to the home folder.
    home_pwd = os.path.abspath("~")

    # Run the main function
    sudo_password = None  # pylint: disable=invalid-name
    asyncio.run(main(args, home_pwd))

    # Return success
    sys.exit(0)
