#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script interactively visualizes trajectories from end_to_end_benchmark.py
output files using Pinocchio and MeshCat.

It allows a user to load a benchmark result file, choose a specific
planning stage and trial, and then inspect the resulting trajectory
frame-by-frame or animate it. This version includes robust mesh path finding
and explicit kinematic updates to prevent rendering artifacts.
"""

# Standard imports
import argparse
import json
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import meshcat
import meshcat.geometry as g
import numpy as np
import pandas as pd
import pinocchio as pin
import pinocchio.visualize


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


def find_ros_package_paths(xacro_file_path: str, logger_func=print) -> List[str]:
    """
    Finds and returns a list of package paths for Pinocchio to search for meshes.
    This function robustly finds the Colcon workspace by searching upwards from
    the provided XACRO file's location.
    """
    package_dirs = []
    ros_package_path = os.environ.get("ROS_PACKAGE_PATH")
    if ros_package_path:
        package_dirs.extend(
            [p for p in ros_package_path.split(os.pathsep) if os.path.isdir(p)]
        )

    current_path = os.path.abspath(os.path.dirname(xacro_file_path))
    ws_root = None
    while True:
        if os.path.isdir(os.path.join(current_path, "src")):
            ws_root = current_path
            break
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            break
        current_path = parent_path

    if ws_root:
        logger_func(f"Found workspace root by searching upwards: {ws_root}")
        src_dir = os.path.join(ws_root, "src")
        if os.path.isdir(src_dir) and src_dir not in package_dirs:
            package_dirs.append(src_dir)
        install_dir = os.path.join(ws_root, "install")
        if os.path.isdir(install_dir) and install_dir not in package_dirs:
            package_dirs.append(install_dir)

    unique_package_dirs = list(dict.fromkeys(package_dirs))
    logger_func(f"Using package search paths for Pinocchio: {unique_package_dirs}")
    return unique_package_dirs


def load_pinocchio_model_from_urdf_string(
    urdf_xml_string: str, package_dirs: List[str], logger_func=print
):
    """Loads a Pinocchio model from a URDF XML string via a temporary file."""
    temp_urdf_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        ) as temp_file:
            temp_urdf_path = temp_file.name
            temp_file.write(urdf_xml_string)

        model = pin.buildModelFromUrdf(temp_urdf_path)
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
                valid_entries = [
                    entry for entry in data if entry.get("trajectory") is not None
                ]
                all_data.extend(valid_entries)
        except Exception as e:
            logger_func(f"Warning: Could not load or parse {file_path}. Error: {e}")

    if not all_data:
        return pd.DataFrame()

    logger_func(
        f"Successfully loaded {len(all_data)} trials with trajectories from {len(file_paths)} file(s)."
    )
    return pd.DataFrame(all_data)


def select_trajectory(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Interactively prompts the user to select a stage and a specific trial."""
    while True:
        stages = df["stage"].unique()
        print("\n--- Please Select a Stage to Visualize ---")
        for i, stage in enumerate(stages):
            print(f"  [{i + 1}] {stage}")
        print("  [q] Quit")

        try:
            choice_str = input(f"Enter choice (1-{len(stages)} or q): ").strip().lower()
            if choice_str == "q":
                return None
            stage_choice = int(choice_str) - 1
            if not 0 <= stage_choice < len(stages):
                raise ValueError
            selected_stage = stages[stage_choice]
        except (ValueError, IndexError):
            print("Invalid choice.")
            continue

        stage_df = df[
            (df["stage"] == selected_stage) & (df["status"] == "Success")
        ].sort_values("trial_id")

        if stage_df.empty:
            print(f"No successful trials found for stage: {selected_stage}")
            continue

        trajectories = stage_df.to_dict("records")
        print(f"\n--- Select a Successful Trial from Stage '{selected_stage}' ---")
        for i, trial in enumerate(trajectories):
            print(
                f"  [{i + 1}] Trial ID: {trial['trial_id']} (IK Attempts: {trial['ik_attempts']})"
            )
        print("  [m] Back to Stage Selection")

        try:
            choice_str = (
                input(f"Enter choice (1-{len(trajectories)} or m): ").strip().lower()
            )
            if choice_str == "m":
                continue
            traj_choice = int(choice_str) - 1
            if not 0 <= traj_choice < len(trajectories):
                raise ValueError
            return trajectories[traj_choice]
        except (ValueError, IndexError):
            print("Invalid choice.")
            continue


def draw_scene_frames(pin_viz, scene_poses: Dict[str, Any], frame_scale: float = 0.2):
    """Draws coordinate frames and identifying markers for all poses in the scene."""
    if not scene_poses:
        return

    print("Drawing scene frames and markers...")
    for name, pose_data in scene_poses.items():
        position = np.array(pose_data["position"])
        quat_xyzw = pose_data["orientation_xyzw"]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        transform = pin.SE3(pin.Quaternion(quat_wxyz), position)

        # Create a parent path for the frame and its marker
        frame_path = f"scene/frames/{name}"

        # Draw the coordinate frame (triad)
        pin_viz.viewer[f"{frame_path}/triad"].set_object(g.triad(frame_scale))

        # Draw a small sphere as a visual marker/label
        # This can be clicked in the viewer to identify the frame by its name
        material = g.MeshLambertMaterial(color=0x5555FF, transparent=True, opacity=0.6)
        pin_viz.viewer[f"{frame_path}/marker"].set_object(g.Sphere(0.02), material)

        # Apply the same transform to the parent path
        pin_viz.viewer[frame_path].set_transform(transform.homogeneous)


def visualization_loop(pin_viz, model, data, selected_trial, args):
    """Runs the interactive UI for a single selected trajectory."""
    # Clear previous frames and draw the ones for the current trial
    try:
        pin_viz.viewer["scene/frames"].delete()
    except KeyError:
        pass  # It's okay if the path doesn't exist on the first run

    if "scene_poses" in selected_trial:
        draw_scene_frames(pin_viz, selected_trial["scene_poses"])

    q = pin.neutral(model)
    trajectory = selected_trial["trajectory"]
    joint_names_from_traj = trajectory["joint_names"]
    waypoints = trajectory["points"]

    joint_map = {
        name: model.getJointId(name)
        for name in joint_names_from_traj
        if model.existJointName(name)
    }

    current_idx = 0
    while True:
        waypoint_positions = waypoints[current_idx]["positions"]
        for name, pos in zip(joint_names_from_traj, waypoint_positions):
            if name in joint_map:
                joint_id = joint_map[name]
                if model.joints[joint_id].nq == 1:
                    q[model.joints[joint_id].idx_q] = pos
                elif model.joints[joint_id].nq == 2:
                    q[model.joints[joint_id].idx_q] = np.cos(pos)
                    q[model.joints[joint_id].idx_q + 1] = np.sin(pos)

        pin.forwardKinematics(model, data, q)
        pin_viz.display(q)

        print(f"\nDisplaying Frame: {current_idx + 1}/{len(waypoints)}")
        print("Commands: [n]ext, [p]rev, [f]irst, [l]ast, [a]nimate, [m]enu, [q]uit")

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
        elif user_input == "a":
            print("Animating... Press Ctrl+C to stop.")
            try:
                for i in range(current_idx, len(waypoints)):
                    waypoint_positions = waypoints[i]["positions"]
                    for name, pos in zip(joint_names_from_traj, waypoint_positions):
                        if name in joint_map:
                            joint_id = joint_map[name]
                            if model.joints[joint_id].nq == 1:
                                q[model.joints[joint_id].idx_q] = pos
                            elif model.joints[joint_id].nq == 2:
                                q[model.joints[joint_id].idx_q] = np.cos(pos)
                                q[model.joints[joint_id].idx_q + 1] = np.sin(pos)

                    pin.forwardKinematics(model, data, q)
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
    """Main execution function."""
    print("--- Interactive Benchmark Trajectory Visualizer ---")

    df = load_benchmark_data(args.benchmark_files)
    if df.empty:
        print("No data with trajectories could be loaded. Exiting.")
        return

    package_dirs = find_ros_package_paths(args.xacro_file)
    urdf_string = xacro_to_urdf_string(args.xacro_file)
    if not urdf_string:
        return

    model, collision_model, visual_model, data = load_pinocchio_model_from_urdf_string(
        urdf_xml_string=urdf_string, package_dirs=package_dirs
    )
    if not model:
        return

    try:
        visualizer = meshcat.Visualizer().open()
        pin_viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
        pin_viz.initViewer(viewer=visualizer)
        pin_viz.loadViewerModel(rootNodeName="robot")
        print(f"\nMeshCat viewer URL: {visualizer.url()}\n")
    except Exception as e:
        print(f"Error initializing MeshCat: {e}")
        return

    while True:
        selected_trial = select_trajectory(df)
        if selected_trial is None:
            break

        action = visualization_loop(pin_viz, model, data, selected_trial, args)
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
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for animation."
    )
    args = parser.parse_args()
    main(args)
