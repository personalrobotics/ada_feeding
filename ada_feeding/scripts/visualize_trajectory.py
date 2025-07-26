#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script interactively visualizes trajectories from holistic_benchmark.py
output files using Pinocchio and MeshCat.

It allows a user to load multiple benchmark result files, choose a specific
planning mode, and then select a successful trajectory from that mode to
inspect frame-by-frame or animate. The script runs in a persistent session,
allowing multiple trajectories to be viewed without relaunching.
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
import pandas as pd

# Third-party imports for saving images/videos
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
    try:
        process = subprocess.run(
            ["ros2", "run", "xacro", "xacro", xacro_filename],
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout
    except Exception as e:
        logger_func(f"An unexpected error occurred during XACRO processing: {e}")
        return None


def find_ros_workspace_root(start_path: str) -> Optional[str]:
    """Searches upwards from a starting path for a directory containing 'src'."""
    current_path = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current_path, "src")):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:  # Reached the filesystem root
            return None
        current_path = parent_path


def load_pinocchio_model_from_urdf_string(
    urdf_xml_string: str, xacro_file_path: str, logger_func=print
):
    """Loads a Pinocchio model from a URDF XML string via a temporary file."""
    temp_urdf_path = ""
    package_dirs = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        ) as temp_file:
            temp_urdf_path = temp_file.name
            temp_file.write(urdf_xml_string)

        # Robustly find package directories for mesh files
        ros_package_path = os.environ.get("ROS_PACKAGE_PATH")
        if ros_package_path:
            package_dirs.extend(
                [p for p in ros_package_path.split(os.pathsep) if os.path.isdir(p)]
            )

        ws_root = find_ros_workspace_root(os.path.dirname(xacro_file_path))
        if ws_root:
            logger_func(f"Found potential workspace root at: {ws_root}")
            src_dir = os.path.join(ws_root, "src")
            if os.path.isdir(src_dir) and src_dir not in package_dirs:
                package_dirs.append(src_dir)
            install_dir = os.path.join(ws_root, "install")
            if os.path.isdir(install_dir) and install_dir not in package_dirs:
                package_dirs.append(install_dir)

        package_dirs = list(dict.fromkeys(package_dirs))
        logger_func(f"Using package search paths for Pinocchio: {package_dirs}")

        # Load the kinematic model from the temporary URDF file path.
        model = pin.buildModelFromUrdf(temp_urdf_path)

        # Load the geometry (visual/collision meshes) using the package search paths.
        visual_model = pin.buildGeomFromUrdf(
            model, temp_urdf_path, pin.GeometryType.VISUAL, package_dirs=package_dirs
        )
        collision_model = pin.buildGeomFromUrdf(
            model, temp_urdf_path, pin.GeometryType.COLLISION, package_dirs=package_dirs
        )

        data = model.createData()
        logger_func(f"Pinocchio model loaded. Nq: {model.nq}, Nv: {model.nv}")
        return model, collision_model, visual_model, data
    except Exception as e:
        logger_func("\n" + "=" * 80)
        logger_func("FATAL: Error loading Pinocchio model.")
        logger_func(
            "This often happens if mesh files (.stl, .dae) referenced in the URDF cannot be found."
        )
        logger_func(
            "Please ensure your ROS 2 workspace is sourced correctly, so that `package://` paths can be resolved."
        )
        logger_func(f"Pinocchio search paths used: {package_dirs}")
        logger_func(f"Original Pinocchio error: {e}")
        logger_func("=" * 80 + "\n")
        return None, None, None, None
    finally:
        if temp_urdf_path and os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)


def load_benchmark_data(file_paths: List[str], logger_func=print) -> pd.DataFrame:
    """Loads and concatenates data from multiple benchmark JSON files."""
    all_data = []
    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            logger_func(f"Warning: Could not load or parse {file_path}. Error: {e}")
    if not all_data:
        return pd.DataFrame()
    logger_func(
        f"Successfully loaded {len(all_data)} total trials from {len(file_paths)} file(s)."
    )
    return pd.json_normalize(all_data, sep="_")


def get_pinocchio_joint_info(
    model: pin.Model, joint_name: str
) -> Optional[Dict[str, Any]]:
    """Gets key info for a named joint in the Pinocchio model."""
    if model.existJointName(joint_name):
        joint_id = model.getJointId(joint_name)
        joint_obj = model.joints[joint_id]
        if joint_obj.idx_q >= 0:
            return {
                "q_idx_start": joint_obj.idx_q,
                "nq": joint_obj.nq,
                "nv": joint_obj.nv,
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
    """Updates Pinocchio q_vector based on a waypoint from the benchmark data."""
    # Set Jaco arm joints
    jaco_positions = waypoint.get("jaco_positions_rad", [])
    for k, mapping_info in enumerate(jaco_mappings):
        if mapping_info and k < len(jaco_positions):
            theta_traj = jaco_positions[k]
            q_idx_start, nq = mapping_info["q_idx_start"], mapping_info["nq"]
            if nq == 1:
                q_vector[q_idx_start] = theta_traj
            elif nq == 2:  # Revolute joint with quaternion representation
                q_vector[q_idx_start], q_vector[q_idx_start + 1] = (
                    np.cos(theta_traj),
                    np.sin(theta_traj),
                )

    # Set Articutool joints
    articutool_solution = waypoint.get("articutool_solution_rad")
    is_feasible = isinstance(articutool_solution, dict)

    if is_rigid or not is_feasible:
        pitch_sol, roll_sol = 0.0, 0.0
    else:
        pitch_sol = articutool_solution.get("pitch", 0.0)
        roll_sol = articutool_solution.get("roll", 0.0)

    for sol, info in [(pitch_sol, atool_pitch_info), (roll_sol, atool_roll_info)]:
        if sol is not None and info:
            q_idx, nq = info["q_idx_start"], info["nq"]
            if nq == 1:
                q_vector[q_idx] = sol
            elif nq == 2:
                q_vector[q_idx], q_vector[q_idx + 1] = np.cos(sol), np.sin(sol)


def select_trajectory(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Interactively prompts the user to select a mode and trajectory."""
    # --- Step 1: Select Planning Mode ---
    modes = df["planning_mode"].unique()
    print("\n--- Please Select a Planning Mode ---")
    for i, mode in enumerate(modes):
        print(f"  [{i + 1}] {mode}")
    print("  [q] Quit")

    try:
        choice_str = input(f"Enter choice (1-{len(modes)} or q): ").strip().lower()
        if choice_str == "q":
            return None
        mode_choice = int(choice_str) - 1
        if not 0 <= mode_choice < len(modes):
            raise ValueError
        selected_mode = modes[mode_choice]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return "retry"  # Special value to retry selection

    # --- Step 2: Select Trajectory from Successful Trials ---
    mode_df = df[(df["planning_mode"] == selected_mode) & (df["status"] == "Success")]
    if mode_df.empty:
        print(f"No successful trials found for mode: {selected_mode}")
        return "retry"

    trajectories = mode_df.sort_values("task_id").to_dict("records")
    print(f"\n--- Select a Successful Trajectory from '{selected_mode}' ---")
    for i, trial in enumerate(trajectories):
        print(
            f"  [{i + 1}] Task ID: {trial['task_id']} (Planning Time: {trial['planning_time_s']:.2f}s)"
        )
    print("  [m] Back to Mode Selection")

    try:
        choice_str = (
            input(f"Enter choice (1-{len(trajectories)} or m): ").strip().lower()
        )
        if choice_str == "m":
            return "retry"
        traj_choice = int(choice_str) - 1
        if not 0 <= traj_choice < len(trajectories):
            raise ValueError
        return trajectories[traj_choice]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return "retry"


def visualization_loop(
    pin_viz, model, waypoints, jaco_mappings, atool_pitch_info, atool_roll_info, args
):
    """Runs the interactive UI for a single selected trajectory."""
    q = pin.neutral(model)
    is_rigid_mode = args.rigid
    current_idx = 0

    while True:
        waypoint_info = waypoints[current_idx]
        set_q_from_waypoint(
            q,
            model,
            waypoint_info,
            jaco_mappings,
            atool_pitch_info,
            atool_roll_info,
            is_rigid_mode,
        )
        pin_viz.display(q)

        is_feasible_in_data = isinstance(
            waypoint_info.get("articutool_solution_rad"), dict
        )
        rigidity_status = (
            "ON (Rigid)"
            if is_rigid_mode
            else f"OFF (Data says feasible: {is_feasible_in_data})"
        )

        print(f"\nDisplaying Frame: {current_idx + 1}/{len(waypoints)}")
        print(f"Articutool Rigidity: {rigidity_status}")
        print(
            "Commands: [n]ext, [p]rev, [f]irst, [l]ast, [t]oggle, [a]nimate, [s]creenshot, [m]enu, [q]uit"
        )

        user_input = input("Enter command: ").strip().lower()

        if user_input == "q":
            return "quit"
        if user_input == "m":
            return "menu"

        if user_input == "n":
            current_idx = min(current_idx + 1, len(waypoints) - 1)
        elif user_input == "p":
            current_idx = max(current_idx - 1, 0)
        elif user_input == "f":
            current_idx = 0
        elif user_input == "l":
            current_idx = len(waypoints) - 1
        elif user_input == "t":
            is_rigid_mode = not is_rigid_mode
            print(f"Articutool rigidity toggled to: {'ON' if is_rigid_mode else 'OFF'}")
        elif user_input == "s":
            print("Taking screenshot...")
            img = visualizer.get_image()
            rigidity_str = "rigid" if is_rigid_mode else "active"
            filepath = os.path.join(
                args.output_dir, f"frame_{current_idx + 1}_{rigidity_str}.png"
            )
            img.save(filepath)
            print(f"Screenshot saved to {filepath}")
        elif user_input == "a":
            print("Animating... Press Ctrl+C to stop.")
            try:
                for i in range(current_idx, len(waypoints)):
                    set_q_from_waypoint(
                        q,
                        model,
                        waypoints[i],
                        jaco_mappings,
                        atool_pitch_info,
                        atool_roll_info,
                        is_rigid_mode,
                    )
                    pin_viz.display(q)
                    print(f"  Displaying frame {i + 1}/{len(waypoints)}", end="\r")
                    time.sleep(1.0 / args.fps)
                current_idx = len(waypoints) - 1
                print("\nAnimation finished.")
            except KeyboardInterrupt:
                print("\nAnimation stopped.")
        elif user_input:
            print("Invalid command.")


def main(args):
    print("--- Interactive Benchmark Trajectory Visualizer ---")

    # --- Load Data and Models (One-time setup) ---
    df = load_benchmark_data(args.benchmark_files)
    if df.empty:
        print("No data could be loaded. Exiting.")
        return

    urdf_string = xacro_to_urdf_string(args.xacro_file)
    if not urdf_string:
        return

    model, collision_model, visual_model, data = load_pinocchio_model_from_urdf_string(
        urdf_string, args.xacro_file
    )
    if not model:
        return

    # --- Initialize Visualizer (One-time setup) ---
    try:
        global visualizer
        visualizer = meshcat.Visualizer().open()
        pin_viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
        pin_viz.initViewer(viewer=visualizer)
        pin_viz.loadViewerModel(rootNodeName=model.name or "robot")
        print(f"MeshCat viewer URL: {visualizer.url()}")
    except Exception as e:
        print(f"Error initializing MeshCat: {e}")
        return

    # --- Main Application Loop ---
    while True:
        selected_trial = select_trajectory(df)

        if selected_trial is None:  # User chose to quit
            break
        if selected_trial == "retry":  # User made an invalid choice or wants to go back
            continue

        waypoints = selected_trial.get("trajectory_metrics_waypoints_data", [])
        if not waypoints:
            print("Selected trial contains no waypoints. Returning to menu.")
            continue

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Screenshots and videos will be saved to: {args.output_dir}")

        jaco_joint_names = [f"{args.prefix}_joint_{i + 1}" for i in range(6)]
        jaco_mappings = [
            get_pinocchio_joint_info(model, name) for name in jaco_joint_names
        ]
        articutool_pitch_info = get_pinocchio_joint_info(
            model, ARTICUTOOL_PITCH_JOINT_NAME
        )
        articutool_roll_info = get_pinocchio_joint_info(
            model, ARTICUTOOL_ROLL_JOINT_NAME
        )

        if (
            not all(jaco_mappings)
            or not articutool_pitch_info
            or not articutool_roll_info
        ):
            print("Error: Could not map all required joints to Pinocchio model.")
            for name, mapping in zip(jaco_joint_names, jaco_mappings):
                if not mapping:
                    print(f"  - FAILED to find joint '{name}' in the Pinocchio model.")
            continue

        # Enter the visualization UI for the selected trajectory
        action = visualization_loop(
            pin_viz,
            model,
            waypoints,
            jaco_mappings,
            articutool_pitch_info,
            articutool_roll_info,
            args,
        )

        if action == "quit":
            break

    print("Visualizer finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize robot trajectories from benchmark files."
    )
    parser.add_argument("xacro_file", type=str, help="Path to the robot XACRO file.")
    parser.add_argument(
        "benchmark_files", nargs="+", help="One or more benchmark JSON result files."
    )
    # --- FIX: Changed default prefix to the correct one ---
    parser.add_argument(
        "--prefix",
        type=str,
        default="j2n6s200",
        help="The prefix for the Jaco arm's joint names (e.g., j2n6s200).",
    )
    parser.add_argument("--rigid", action="store_true", help="Start in rigid mode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualization_output",
        help="Directory to save screenshots and videos.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for animation."
    )
    args = parser.parse_args()
    main(args)
