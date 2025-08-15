#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script interactively visualizes trajectories from end_to_end_benchmark.py
output files using Pinocchio and MeshCat.

It allows a user to load a benchmark result file, choose a specific
planning stage and trial, and then inspect the resulting trajectory
frame-by-frame or animate it.

This version automatically handles different execution modes (sequential, synchronous)
and includes robust mesh path finding and explicit kinematic updates to prevent
rendering artifacts.
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
    """
    Loads and flattens data from one or more benchmark JSON files into a pandas DataFrame.
    The new hierarchical structure (trial -> stages) is flattened so that each
    stage with a valid trajectory becomes a single row in the DataFrame.
    """
    all_stages_flat = []
    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                trials = json.load(f)
                for trial in trials:
                    for stage in trial.get("stages", []):
                        flat_record = {
                            "trial_id": trial.get("trial_id"),
                            "scene_poses": trial.get("scene_poses"),
                            "stage_name": stage.get("stage_name"),
                            "status": stage.get("status"),
                            "execution_mode": stage.get("execution_mode"),
                            "traj_jaco": stage.get("traj_jaco"),
                            "traj_atool": stage.get("traj_atool"),
                        }
                        if flat_record["traj_jaco"] or flat_record["traj_atool"]:
                            all_stages_flat.append(flat_record)
        except Exception as e:
            logger_func(f"Warning: Could not load or parse {file_path}. Error: {e}")

    if not all_stages_flat:
        return pd.DataFrame()

    logger_func(f"Successfully loaded {len(all_stages_flat)} stages with trajectories.")
    return pd.DataFrame(all_stages_flat)


def select_trajectory(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Interactively prompts the user to select a stage and a trial. It then finds
    the static pose of the Jaco arm from the previous stage to ensure
    visual continuity.
    """
    stage_order = [
        "HomeToAbovePlate",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelArticutool",
    ]

    while True:
        stages = df["stage_name"].unique()
        print("\n--- Please Select a Stage to Visualize ---")
        for i, stage in enumerate(stages):
            print(f"  [{i + 1}] {stage}")
        print("  [q] Quit")
        try:
            choice_str = input(f"Enter choice (1-{len(stages)} or q): ").strip().lower()
            if choice_str == "q":
                return None
            selected_stage_name = stages[int(choice_str) - 1]
        except (ValueError, IndexError):
            print("Invalid choice.")
            continue

        stage_df = df[
            (df["stage_name"] == selected_stage_name) & (df["status"] == "Success")
        ].sort_values("trial_id")

        if stage_df.empty:
            print(f"No successful trials found for stage: {selected_stage_name}")
            continue

        trials_for_stage = stage_df.to_dict("records")
        print(f"\n--- Select a Successful Trial from Stage '{selected_stage_name}' ---")
        for i, trial in enumerate(trials_for_stage):
            print(
                f"  [{i + 1}] Trial ID: {trial['trial_id']} (Mode: {trial['execution_mode']})"
            )
        print("  [m] Back to Stage Selection")
        try:
            choice_str = (
                input(f"Enter choice (1-{len(trials_for_stage)} or m): ")
                .strip()
                .lower()
            )
            if choice_str == "m":
                continue

            selected_record = trials_for_stage[int(choice_str) - 1]
            trial_id = selected_record["trial_id"]

            try:
                current_stage_index = stage_order.index(selected_stage_name)
                if current_stage_index > 0:
                    previous_stage_name = stage_order[current_stage_index - 1]
                    previous_stage_df = df[
                        (df["trial_id"] == trial_id)
                        & (df["stage_name"] == previous_stage_name)
                    ]
                    if not previous_stage_df.empty:
                        prev_traj_jaco = previous_stage_df.iloc[0]["traj_jaco"]
                        if prev_traj_jaco and prev_traj_jaco["points"]:
                            last_waypoint = prev_traj_jaco["points"][-1]
                            selected_record["static_jaco_config"] = {
                                "positions": last_waypoint["positions"],
                                "joint_names": prev_traj_jaco["joint_names"],
                            }
            except ValueError:
                print(
                    f"Warning: Stage '{selected_stage_name}' not in defined stage order for state carryover."
                )

            return selected_record
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
        quat_xyzw = np.array(pose_data["orientation_xyzw"])

        transform = pin.SE3(pin.Quaternion(quat_xyzw), position)

        # Create a parent path for the frame and its marker
        frame_path = f"scene/frames/{name}"

        # Draw the coordinate frame (triad)
        pin_viz.viewer[f"{frame_path}/triad"].set_object(g.triad(frame_scale))

        # Draw a small sphere as a visual marker/label
        material = g.MeshLambertMaterial(color=0x5555FF, transparent=True, opacity=0.6)
        pin_viz.viewer[f"{frame_path}/marker"].set_object(g.Sphere(0.02), material)

        # Apply the same transform to the parent path
        pin_viz.viewer[frame_path].set_transform(transform.homogeneous)


def get_joint_map(model, joint_names):
    """Helper to map joint names to their indices in Pinocchio's q vector."""
    if not joint_names:
        return {}
    return {
        name: model.getJointId(name)
        for name in joint_names
        if model.existJointName(name)
    }


def update_q_from_waypoint(q, model, waypoint, joint_names, joint_map):
    """Helper to update the robot configuration from a single waypoint."""
    if not waypoint:
        return
    for name, pos in zip(joint_names, waypoint["positions"]):
        if name in joint_map:
            joint_id = joint_map[name]
            if model.joints[joint_id].nq == 1:
                q[model.joints[joint_id].idx_q] = pos
            elif model.joints[joint_id].nq == 2:
                q[model.joints[joint_id].idx_q] = np.cos(pos)
                q[model.joints[joint_id].idx_q + 1] = np.sin(pos)


def visualization_loop(pin_viz, model, data, selected_stage, args):
    """Runs the interactive UI, handling different execution modes and frame-by-frame controls."""
    try:
        pin_viz.viewer["scene/frames"].delete()
    except KeyError:
        pass
    if "scene_poses" in selected_stage:
        draw_scene_frames(pin_viz, selected_stage["scene_poses"])

    execution_mode = selected_stage["execution_mode"]
    traj_jaco = selected_stage.get("traj_jaco")
    traj_atool = selected_stage.get("traj_atool")
    static_jaco_config = selected_stage.get("static_jaco_config")

    jaco_joint_map = get_joint_map(
        model, (traj_jaco or static_jaco_config or {}).get("joint_names", [])
    )
    atool_joint_map = get_joint_map(model, (traj_atool or {}).get("joint_names", []))

    # --- Prepare a unified list of waypoints for consistent navigation ---
    waypoints = []
    if execution_mode in ["Jaco Only", "Articutool Only"]:
        traj = traj_jaco if execution_mode == "Jaco Only" else traj_atool
        waypoints = traj["points"]
    elif execution_mode == "Sequential":
        waypoints.extend(traj_jaco["points"])
        waypoints.extend(traj_atool["points"])
    elif execution_mode == "Synchronous":
        waypoints = traj_jaco["points"] if traj_jaco else []  # Default to Jaco

    if not waypoints:
        print("No waypoints to display for this stage.")
        return "menu"

    current_idx = 0
    q = pin.neutral(model)

    while True:
        # --- Update robot configuration `q` for the current frame ---
        # 1. Start with the static Jaco pose from the previous stage, if available.
        if static_jaco_config:
            update_q_from_waypoint(
                q,
                model,
                {"positions": static_jaco_config["positions"]},
                static_jaco_config["joint_names"],
                jaco_joint_map,
            )

        # 2. Layer on the animated joint positions for the current waypoint.
        if execution_mode == "Sequential":
            # For sequential, we need to know the final pose of the first trajectory
            jaco_end_waypoint = traj_jaco["points"][-1]
            if current_idx >= len(traj_jaco["points"]):  # We are in the Articutool part
                update_q_from_waypoint(
                    q,
                    model,
                    jaco_end_waypoint,
                    traj_jaco["joint_names"],
                    jaco_joint_map,
                )
                update_q_from_waypoint(
                    q,
                    model,
                    waypoints[current_idx],
                    traj_atool["joint_names"],
                    atool_joint_map,
                )
            else:  # We are in the Jaco part
                update_q_from_waypoint(
                    q,
                    model,
                    waypoints[current_idx],
                    traj_jaco["joint_names"],
                    jaco_joint_map,
                )
                # Keep atool at its starting position
                update_q_from_waypoint(
                    q,
                    model,
                    traj_atool["points"][0],
                    traj_atool["joint_names"],
                    atool_joint_map,
                )
        else:  # For all other modes, just update from the single relevant trajectory
            if traj_jaco:
                update_q_from_waypoint(
                    q,
                    model,
                    traj_jaco["points"][min(current_idx, len(traj_jaco["points"]) - 1)],
                    traj_jaco["joint_names"],
                    jaco_joint_map,
                )
            if traj_atool:
                update_q_from_waypoint(
                    q,
                    model,
                    traj_atool["points"][
                        min(current_idx, len(traj_atool["points"]) - 1)
                    ],
                    traj_atool["joint_names"],
                    atool_joint_map,
                )

        pin.forwardKinematics(model, data, q)
        pin_viz.display(q)

        print(f"\nDisplaying Frame: {current_idx + 1}/{len(waypoints)}")
        print(
            f"--- Stage: {selected_stage['stage_name']} | Trial: {selected_stage['trial_id']} | Mode: {execution_mode} ---"
        )
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
                    # The same update logic as above is used for animation frames
                    if static_jaco_config:
                        update_q_from_waypoint(
                            q,
                            model,
                            {"positions": static_jaco_config["positions"]},
                            static_jaco_config["joint_names"],
                            jaco_joint_map,
                        )
                    if execution_mode == "Sequential":
                        jaco_end_waypoint = traj_jaco["points"][-1]
                        if i >= len(traj_jaco["points"]):
                            update_q_from_waypoint(
                                q,
                                model,
                                jaco_end_waypoint,
                                traj_jaco["joint_names"],
                                jaco_joint_map,
                            )
                            update_q_from_waypoint(
                                q,
                                model,
                                waypoints[i],
                                traj_atool["joint_names"],
                                atool_joint_map,
                            )
                        else:
                            update_q_from_waypoint(
                                q,
                                model,
                                waypoints[i],
                                traj_jaco["joint_names"],
                                jaco_joint_map,
                            )
                            update_q_from_waypoint(
                                q,
                                model,
                                traj_atool["points"][0],
                                traj_atool["joint_names"],
                                atool_joint_map,
                            )
                    else:
                        if traj_jaco:
                            update_q_from_waypoint(
                                q,
                                model,
                                traj_jaco["points"][
                                    min(i, len(traj_jaco["points"]) - 1)
                                ],
                                traj_jaco["joint_names"],
                                jaco_joint_map,
                            )
                        if traj_atool:
                            update_q_from_waypoint(
                                q,
                                model,
                                traj_atool["points"][
                                    min(i, len(traj_atool["points"]) - 1)
                                ],
                                traj_atool["joint_names"],
                                atool_joint_map,
                            )

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
        selected_stage = select_trajectory(df)
        if selected_stage is None:
            break
        action = visualization_loop(pin_viz, model, data, selected_stage, args)
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
