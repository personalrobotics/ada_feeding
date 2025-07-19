#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script is used to visualize robot trajectories or single joint states
using Pinocchio and MeshCat. It can load:
1. Enhanced trajectory JSON files (containing Jaco and Articutool waypoints).
2. Single joint state JSON files (representing a sensor_msgs/msg/JointState).

New features for figure generation:
--snapshot-frames: Enters a mode to navigate only specific keyframes.
--rigid: Sets the initial state of the Articutool's rigidity.
't' command: Toggles Articutool rigidity on/off interactively.
's' command: Saves a labeled screenshot of the current view.

New features for video recording:
'r' command: Starts and stops recording the current view. The user can move
             the camera in MeshCat while recording.
'a' command: Now prompts to record the animation to a file.
--fps: Sets the frames per second for the output video.
"""

# Standard imports
import pinocchio as pin
import pinocchio.visualize
import meshcat
import numpy as np
import json
import subprocess
import tempfile
import os
import argparse
import time
import sys
import math
from typing import Optional, Dict, Any, List

# --- MODIFICATION: Added Pillow for image saving and imageio for video ---
# Note: You may need to install imageio and a backend:
# pip install imageio imageio-ffmpeg
from PIL import Image
import imageio


# Define Articutool joint names as they appear in the Pinocchio model
ARTICUTOOL_PITCH_JOINT_NAME = "atool_joint1"
ARTICUTOOL_ROLL_JOINT_NAME = "atool_joint2"


def xacro_to_urdf_string(xacro_filename: str, logger_func=print) -> Optional[str]:
    """Converts a Xacro file to a URDF XML string using ros2 run xacro."""
    if not os.path.exists(xacro_filename):
        logger_func(f"Error: Xacro file not found at {xacro_filename}")
        return None

    logger_func(f"Processing Xacro file: {xacro_filename}")
    try:
        process = subprocess.run(
            ["ros2", "run", "xacro", "xacro", xacro_filename],
            check=True,
            capture_output=True,
            text=True,
        )
        urdf_xml_string = process.stdout
        logger_func("XACRO processing successful.")
        return urdf_xml_string
    except FileNotFoundError:
        logger_func(
            "Fatal: Command 'ros2' not found. Make sure ROS 2 environment is sourced."
        )
        return None
    except subprocess.CalledProcessError as e:
        logger_func(
            f"Fatal: XACRO processing command failed with exit code {e.returncode}."
        )
        logger_func(f"XACRO stdout:\n{e.stdout}")
        logger_func(f"XACRO stderr:\n{e.stderr}")
        return None
    except Exception as e:
        logger_func(f"An unexpected error occurred during XACRO processing: {e}")
        return None


def load_pinocchio_model_from_urdf_string(urdf_xml_string: str, logger_func=print):
    """Loads a Pinocchio model from a URDF XML string via a temporary file."""
    temp_urdf_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        ) as temp_file:
            temp_urdf_path = temp_file.name
            temp_file.write(urdf_xml_string)
        logger_func(f"Generated temporary URDF file: {temp_urdf_path}")

        package_dirs = []
        ros_package_path = os.environ.get("ROS_PACKAGE_PATH")
        if ros_package_path:
            package_dirs = [
                p for p in ros_package_path.split(os.pathsep) if os.path.isdir(p)
            ]

        common_workspace_paths = [
            os.path.expanduser("~/ros2_ws/src"),
            os.path.expanduser("~/dev_ws/src"),
            os.path.expanduser("~/workspace/src"),
        ]
        for wsp in common_workspace_paths:
            if os.path.isdir(wsp) and wsp not in package_dirs:
                package_dirs.append(wsp)

        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):
            if os.path.basename(current_dir) in ["install", "build", "log"]:
                ws_root = os.path.dirname(current_dir)
                src_path = os.path.join(ws_root, "src")
                if os.path.isdir(src_path) and src_path not in package_dirs:
                    package_dirs.append(src_path)
                break
            current_dir = os.path.dirname(current_dir)

        if package_dirs:
            unique_package_dirs = list(set(package_dirs))
            logger_func(f"Using package_dirs for Pinocchio: {unique_package_dirs}")
            model = pin.buildModelFromUrdf(
                temp_urdf_path, package_dirs=unique_package_dirs
            )
            collision_model = pin.buildGeomFromUrdf(
                model,
                temp_urdf_path,
                pin.GeometryType.COLLISION,
                package_dirs=unique_package_dirs,
            )
            visual_model = pin.buildGeomFromUrdf(
                model,
                temp_urdf_path,
                pin.GeometryType.VISUAL,
                package_dirs=unique_package_dirs,
            )
        else:
            logger_func(
                "Warning: ROS_PACKAGE_PATH not found or empty. Mesh loading might fail."
            )
            model = pin.buildModelFromUrdf(temp_urdf_path)
            collision_model = pin.buildGeomFromUrdf(
                model, temp_urdf_path, pin.GeometryType.COLLISION
            )
            visual_model = pin.buildGeomFromUrdf(
                model, temp_urdf_path, pin.GeometryType.VISUAL
            )

        data = model.createData()
        logger_func(
            f"Pinocchio model loaded. Name: {model.name}, Nq: {model.nq}, Nv: {model.nv}, NJoints: {model.njoints}"
        )
        return model, collision_model, visual_model, data
    except Exception as e:
        logger_func(f"Error loading Pinocchio model: {e}")
        return None, None, None, None
    finally:
        if temp_urdf_path and os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)


def load_json_file(filepath: str, logger_func=print) -> Optional[Dict[str, Any]]:
    """Loads data from a generic JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger_func(f"Successfully loaded JSON from {filepath}")
        return data
    except FileNotFoundError:
        logger_func(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        logger_func(f"Error: Could not decode JSON from {filepath}")
        return None
    return None


def get_pinocchio_joint_info(
    model: pin.Model, joint_name: str
) -> Optional[Dict[str, Any]]:
    """Gets q_idx_start, nq, and nv for a named joint in the Pinocchio model."""
    if model.existJointName(joint_name):
        joint_id = model.getJointId(joint_name)
        joint_obj = model.joints[joint_id]
        if joint_obj.idx_q >= 0 and joint_obj.nq > 0:
            return {
                "name": joint_name,
                "q_idx_start": joint_obj.idx_q,
                "nq": joint_obj.nq,
                "nv": joint_obj.nv,
                "id": joint_id,
            }
    return None


def set_q_from_waypoint(
    q_vector: np.ndarray,
    model: pin.Model,
    waypoint: Dict[str, Any],
    jaco_mappings,
    atool_pitch_info,
    atool_roll_info,
    is_rigid: bool,
):
    """Updates Pinocchio q_vector based on a waypoint from the trajectory data."""
    jaco_positions = waypoint.get("jaco_positions_rad", [])
    for k, mapping_info in enumerate(jaco_mappings):
        if mapping_info and k < len(jaco_positions):
            theta_traj = jaco_positions[k]
            q_idx_start, nq, nv = (
                mapping_info["q_idx_start"],
                mapping_info["nq"],
                mapping_info["nv"],
            )
            if nq == 1:
                q_vector[q_idx_start] = theta_traj
            elif nq == 2 and nv == 1:
                q_vector[q_idx_start], q_vector[q_idx_start + 1] = (
                    np.cos(theta_traj),
                    np.sin(theta_traj),
                )

    is_feasible = waypoint.get("articutool_waypoint_feasible", False)
    if is_rigid or not is_feasible:
        pitch_sol, roll_sol = 0.0, 0.0
    else:
        pitch_sol = waypoint.get("articutool_pitch_solution_rad", 0.0)
        roll_sol = waypoint.get("articutool_roll_solution_rad", 0.0)

    for sol, info in [(pitch_sol, atool_pitch_info), (roll_sol, atool_roll_info)]:
        if sol is not None and info:
            q_idx, nq, nv = info["q_idx_start"], info["nq"], info["nv"]
            if nq == 1:
                q_vector[q_idx] = sol
            elif nq == 2 and nv == 1:
                q_vector[q_idx], q_vector[q_idx + 1] = np.cos(sol), np.sin(sol)


def main(args):
    print("--- Pinocchio + MeshCat Visualizer ---")

    urdf_string = xacro_to_urdf_string(args.xacro_file)
    if not urdf_string:
        return

    model, collision_model, visual_model, data = load_pinocchio_model_from_urdf_string(
        urdf_string
    )
    if not model:
        return

    print("Initializing MeshCat viewer...")
    try:
        visualizer = meshcat.Visualizer().open()
        print(f"MeshCat viewer URL: {visualizer.url()}")
        pin_viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
        pin_viz.initViewer(viewer=visualizer)
        pin_viz.loadViewerModel(rootNodeName=model.name or "robot")
        print("MeshCat viewer initialized.")
    except Exception as e:
        print(f"Error initializing MeshCat: {e}")
        return

    if not args.trajectory_file:
        print("No trajectory file provided. Displaying neutral pose.")
        pin_viz.display(pin.neutral(model))
        input("Press Enter to quit.")
        return

    trajectory_data = load_json_file(args.trajectory_file)
    if (
        not trajectory_data
        or "jaco_joint_names" not in trajectory_data
        or "waypoints" not in trajectory_data
    ):
        print("Failed to load valid trajectory data. Exiting.")
        return

    waypoints = trajectory_data["waypoints"]
    if not waypoints:
        print("Trajectory contains no waypoints. Exiting.")
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Screenshots and videos will be saved to: {args.output_dir}")

    jaco_joint_names = trajectory_data["jaco_joint_names"]
    jaco_mappings = [get_pinocchio_joint_info(model, name) for name in jaco_joint_names]
    articutool_pitch_info = get_pinocchio_joint_info(model, ARTICUTOOL_PITCH_JOINT_NAME)
    articutool_roll_info = get_pinocchio_joint_info(model, ARTICUTOOL_ROLL_JOINT_NAME)

    if not all(jaco_mappings) or not articutool_pitch_info or not articutool_roll_info:
        print("Error: Could not map all required joints to Pinocchio model.")
        return

    q = pin.neutral(model)

    is_rigid_mode = args.rigid
    navigable_indices = []
    if args.snapshot_frames:
        navigable_indices = [
            int(f.strip()) - 1 for f in args.snapshot_frames.split(",")
        ]
        print(f"\n--- Snapshot Navigation Mode ---")
        print(
            f"Navigating {len(navigable_indices)} keyframes: {[f + 1 for f in navigable_indices]}"
        )
    else:
        navigable_indices = list(range(len(waypoints)))
        print(f"\n--- Full Trajectory Navigation Mode ---")

    current_nav_idx = 0

    is_recording = False
    video_writer = None
    current_recording_filepath = None

    try:
        while True:
            if not navigable_indices:
                print("No frames to navigate.")
                break

            current_point_idx = navigable_indices[current_nav_idx]
            waypoint_info = waypoints[current_point_idx]

            set_q_from_waypoint(
                q,
                model,
                waypoint_info,
                jaco_mappings,
                articutool_pitch_info,
                articutool_roll_info,
                is_rigid_mode,
            )
            pin_viz.display(q)

            is_atool_active_in_file = waypoint_info.get(
                "articutool_waypoint_feasible", False
            )
            rigidity_status = (
                "ON (Rigid)"
                if is_rigid_mode
                else f"OFF (Active: {is_atool_active_in_file})"
            )

            print(
                f"\nDisplaying Frame: {current_point_idx + 1}/{len(waypoints)} (Nav index: {current_nav_idx + 1}/{len(navigable_indices)})"
            )
            print(f"Articutool Rigidity: {rigidity_status}")
            print(
                "Commands: [n]ext, [p]rev, [f]irst, [l]ast, [t]oggle rigidity, [a]nimate, [s]creenshot, [r]ecord, [q]uit"
            )

            user_input = input("Enter command: ").strip().lower()

            if user_input == "q":
                break
            elif user_input == "s":
                print("Taking screenshot...")
                img = visualizer.get_image()
                rigidity_str = "rigid" if is_rigid_mode else "active"
                filename = f"frame_{current_point_idx + 1}_{rigidity_str}.png"
                filepath = os.path.join(args.output_dir, filename)
                img.save(filepath)
                print(f"Screenshot saved to {filepath}")
                continue
            elif user_input == "r":
                if not is_recording:
                    is_recording = True
                    rigidity_str = "rigid" if is_rigid_mode else "active"
                    filename = (
                        f"recording_frame_{current_point_idx + 1}_{rigidity_str}.mp4"
                    )
                    current_recording_filepath = os.path.join(args.output_dir, filename)
                    print(
                        f"Starting recording... Output file: {current_recording_filepath}"
                    )
                    # --- MODIFICATION: Explicitly set the format to 'FFMPEG' to avoid using the wrong writer ---
                    video_writer = imageio.get_writer(
                        current_recording_filepath, fps=args.fps, format="FFMPEG"
                    )

                    print(
                        "Recording... Move the camera in MeshCat. Press Ctrl+C to stop."
                    )
                    try:
                        while True:
                            img = visualizer.get_image()
                            video_writer.append_data(np.array(img))
                            time.sleep(1.0 / args.fps)
                    except KeyboardInterrupt:
                        print("\nStopping manual recording.")
                else:
                    pass

                if video_writer:
                    video_writer.close()
                    print(f"Video saved to {current_recording_filepath}")
                    video_writer = None
                    current_recording_filepath = None
                is_recording = False
                continue

            elif user_input == "t":
                is_rigid_mode = not is_rigid_mode
                print(
                    f"Articutool rigidity toggled to: {'ON' if is_rigid_mode else 'OFF'}"
                )
                continue
            elif user_input == "n":
                current_nav_idx = min(current_nav_idx + 1, len(navigable_indices) - 1)
            elif user_input == "p":
                current_nav_idx = max(current_nav_idx - 1, 0)
            elif user_input == "f":
                current_nav_idx = 0
            elif user_input == "l":
                current_nav_idx = len(navigable_indices) - 1
            elif user_input == "a":
                record_anim = (
                    input("Record this animation? (y/n): ").lower().strip() == "y"
                )
                anim_writer = None
                filepath = ""
                if record_anim:
                    rigidity_str = "rigid" if is_rigid_mode else "active"
                    filename = f"animation_{rigidity_str}.mp4"
                    filepath = os.path.join(args.output_dir, filename)
                    # --- MODIFICATION: Explicitly set the format to 'FFMPEG' to avoid using the wrong writer ---
                    anim_writer = imageio.get_writer(
                        filepath, fps=args.fps, format="FFMPEG"
                    )
                    print(f"Recording animation to {filepath}...")

                print("Animating... Press Ctrl+C to stop.")
                anim_q = q.copy()
                start_anim_nav_idx = current_nav_idx
                try:
                    for i in range(start_anim_nav_idx, len(navigable_indices)):
                        anim_point_idx = navigable_indices[i]
                        set_q_from_waypoint(
                            anim_q,
                            model,
                            waypoints[anim_point_idx],
                            jaco_mappings,
                            articutool_pitch_info,
                            articutool_roll_info,
                            is_rigid_mode,
                        )
                        pin_viz.display(anim_q)

                        if anim_writer:
                            img = visualizer.get_image()
                            anim_writer.append_data(np.array(img))

                        print(
                            f"  Displaying frame {anim_point_idx + 1}/{len(waypoints)}",
                            end="\r",
                        )
                        time.sleep(1.0 / args.fps)
                    current_nav_idx = len(navigable_indices) - 1
                    print("\nAnimation finished.")
                except KeyboardInterrupt:
                    print("\nAnimation stopped.")
                finally:
                    if anim_writer:
                        anim_writer.close()
                        print(f"Animation video saved to {filepath}")

            else:
                if user_input:
                    print("Invalid command.")
                continue

    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
    finally:
        if video_writer:
            video_writer.close()
            if current_recording_filepath:
                print(f"Final video saved to {current_recording_filepath}")
        print("Visualizer finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize robot trajectories using Pinocchio and MeshCat."
    )
    parser.add_argument("xacro_file", type=str, help="Path to the robot XACRO file.")
    parser.add_argument(
        "--trajectory_file",
        type=str,
        help="Path to the .json trajectory file.",
    )
    parser.add_argument(
        "--rigid",
        action="store_true",
        help="Sets the initial state to be rigid. Can be toggled interactively.",
    )
    parser.add_argument(
        "--snapshot-frames",
        type=str,
        help="Comma-separated list of 1-based frame numbers to navigate (e.g., '1,25,50').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="screenshots",
        help="Directory to save screenshots and videos. Default: 'screenshots'",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video recording. Default: 30",
    )

    args = parser.parse_args()
    main(args)
