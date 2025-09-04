#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script interactively visualizes trajectories from end_to_end_benchmark.py
output files using Pinocchio and MeshCat.

This refactored version uses a "trial-centric" workflow. A user first selects
a Trial ID, which immediately displays the static scene for that trial. The user
can then choose to play back any of the available (successful) stage
trajectories from within that trial's context.
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
    Loads data from benchmark .jsonl files, ensuring all trials and
    stages are loaded.
    """
    all_stages_flat = []
    for file_path in file_paths:
        try:
            # --- Read the file line by line for .jsonl format ---
            with open(file_path, "r") as f:
                # Create a list of trial dictionaries by parsing each line
                trials = [json.loads(line) for line in f if line.strip()]

            for trial in trials:
                if not trial.get("stages"):
                    all_stages_flat.append(
                        {
                            "trial_id": trial.get("trial_id"),
                            "scene_poses": trial.get("scene_poses"),
                            "stage_name": "N/A",
                            "status": "NoStages",
                        }
                    )
                    continue

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
                    all_stages_flat.append(flat_record)
        except Exception as e:
            logger_func(f"Warning: Could not load or parse {file_path}. Error: {e}")

    if not all_stages_flat:
        return pd.DataFrame()

    logger_func(
        f"Successfully loaded {len(all_stages_flat)} stage records across all trials."
    )
    return pd.DataFrame(all_stages_flat)


def select_trial(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Prompts the user to select a Trial ID, showing the last successfully
    completed stage for each trial.
    """
    all_trial_ids = sorted(df["trial_id"].unique())
    if not all_trial_ids:
        print("No trials found in the loaded data.")
        return None

    # Pre-calculate the completion status for each trial
    trial_completion_info = {}
    stage_order = [
        "HomeToAbovePlate",
        "PreAcquisition",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelTool",
        "Resting",
        "Staging",
        "Presentation",
    ]
    for trial_id in all_trial_ids:
        trial_df = df[df["trial_id"] == trial_id]
        successful_stages = trial_df[trial_df["status"] == "Success"][
            "stage_name"
        ].tolist()

        last_success = "None"
        last_success_idx = -1
        for stage_name in successful_stages:
            if stage_name in stage_order:
                last_success_idx = max(last_success_idx, stage_order.index(stage_name))

        if last_success_idx != -1:
            last_success = stage_order[last_success_idx]
        trial_completion_info[trial_id] = last_success

    while True:
        print("\n--- Please Select a Trial to Visualize ---")
        for i, trial_id in enumerate(all_trial_ids):
            completion_status = trial_completion_info.get(trial_id, "None")
            print(
                f"  [{i + 1}] Trial ID: {trial_id} (Last Success: {completion_status})"
            )
        print("  [q] Quit")
        try:
            choice_str = (
                input(f"Enter choice (1-{len(all_trial_ids)} or q): ").strip().lower()
            )
            if choice_str == "q":
                return None
            selected_trial_id = all_trial_ids[int(choice_str) - 1]
            return df[df["trial_id"] == selected_trial_id].copy()
        except (ValueError, IndexError):
            print("Invalid choice.")


def draw_scene_frames(pin_viz, scene_poses: Dict[str, Any], frame_scale: float = 0.2):
    """Draws coordinate frames and identifying markers for all poses in the scene."""
    if not scene_poses:
        return

    # Clear any previous frames before drawing new ones
    try:
        pin_viz.viewer["scene/frames"].delete()
    except KeyError:
        pass
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


# --- Helper functions for robot state updates ---
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


def trajectory_playback_loop(pin_viz, model, data, selected_stage, args):
    """
    Runs the interactive UI for a SINGLE trajectory.
    Handles different execution modes and frame-by-frame controls.
    """
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
        waypoints = (traj_jaco or traj_atool).get("points", [])
    elif execution_mode == "Sequential" and traj_jaco and traj_atool:
        waypoints.extend(traj_jaco["points"])
        waypoints.extend(traj_atool["points"])
    elif execution_mode == "Synchronous" and traj_jaco:
        waypoints = traj_jaco["points"]

    if not waypoints:
        print("No waypoints to display for this stage.")
        input("Press Enter to continue...")
        return "back"

    current_idx = 0

    while True:
        # This ensures a clean slate and prevents state from bleeding over.
        q = pin.neutral(model)

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
            if current_idx >= len(traj_jaco["points"]):
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
            else:
                update_q_from_waypoint(
                    q,
                    model,
                    waypoints[current_idx],
                    traj_jaco["joint_names"],
                    jaco_joint_map,
                )
                if traj_atool and traj_atool["points"]:
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

        # 3. Update and display the final calculated pose
        pin.forwardKinematics(model, data, q)
        pin_viz.display(q)

        print(f"\nDisplaying Frame: {current_idx + 1}/{len(waypoints)}")
        print(f"--- Stage: {selected_stage['stage_name']} ---")
        print(
            "Commands: [n]ext, [p]rev, [f]irst, [l]ast, [a]nimate, [b]ack to stage selection, [q]uit"
        )
        user_input = input("Enter command: ").strip().lower()

        if user_input == "q":
            return "quit"
        if user_input == "b":
            return "back"

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
                    # Also reset q inside the animation loop for consistency
                    q_anim = pin.neutral(model)
                    if static_jaco_config:
                        update_q_from_waypoint(
                            q_anim,
                            model,
                            {"positions": static_jaco_config["positions"]},
                            static_jaco_config["joint_names"],
                            jaco_joint_map,
                        )

                    if execution_mode == "Sequential":
                        jaco_end_waypoint = traj_jaco["points"][-1]
                        if i >= len(traj_jaco["points"]):
                            update_q_from_waypoint(
                                q_anim,
                                model,
                                jaco_end_waypoint,
                                traj_jaco["joint_names"],
                                jaco_joint_map,
                            )
                            update_q_from_waypoint(
                                q_anim,
                                model,
                                waypoints[i],
                                traj_atool["joint_names"],
                                atool_joint_map,
                            )
                        else:
                            update_q_from_waypoint(
                                q_anim,
                                model,
                                waypoints[i],
                                traj_jaco["joint_names"],
                                jaco_joint_map,
                            )
                            if traj_atool and traj_atool["points"]:
                                update_q_from_waypoint(
                                    q_anim,
                                    model,
                                    traj_atool["points"][0],
                                    traj_atool["joint_names"],
                                    atool_joint_map,
                                )
                    else:
                        if traj_jaco:
                            update_q_from_waypoint(
                                q_anim,
                                model,
                                traj_jaco["points"][
                                    min(i, len(traj_jaco["points"]) - 1)
                                ],
                                traj_jaco["joint_names"],
                                jaco_joint_map,
                            )
                        if traj_atool:
                            update_q_from_waypoint(
                                q_anim,
                                model,
                                traj_atool["points"][
                                    min(i, len(traj_atool["points"]) - 1)
                                ],
                                traj_atool["joint_names"],
                                atool_joint_map,
                            )

                    pin.forwardKinematics(model, data, q_anim)
                    pin_viz.display(q_anim)
                    print(f"  Displaying frame {i + 1}/{len(waypoints)}", end="\r")
                    time.sleep(1.0 / args.fps)
                current_idx = len(waypoints) - 1
                print("\nAnimation finished.")
            except KeyboardInterrupt:
                print("\nAnimation stopped.")


def trial_visualization_loop(pin_viz, model, data, trial_df, args):
    """
    The main menu loop for an individual trial. Handles trials with
    no successful stages gracefully.
    """
    trial_id = trial_df.iloc[0]["trial_id"]
    scene_poses = trial_df.iloc[0]["scene_poses"]

    # 1. Immediately visualize the scene for this trial
    draw_scene_frames(pin_viz, scene_poses)

    # 2. Prepare a map of all trajectories in this trial for state carry-over
    stage_data_map = {row["stage_name"]: row for row in trial_df.to_dict("records")}
    stage_order = [
        "HomeToAbovePlate",
        "PreAcquisition",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelTool",
        "Resting",
        "Staging",
        "Presentation",
    ]

    while True:
        print(f"\n--- Visualizing Trial ID: {trial_id} ---")
        print("Scene frames are now displayed.")

        available_stages = [
            row for row in trial_df.to_dict("records") if row["status"] == "Success"
        ]

        if not available_stages:
            print("No successful trajectories available to visualize for this trial.")
            print("  [m] Back to Trial Selection")
            print("  [q] Quit")
            choice_str = input(f"Enter choice (m/q): ").strip().lower()
            if choice_str == "q":
                return "quit"
            if choice_str == "m":
                return "menu"
            print("Invalid choice.")
            continue

        print("Select an option:")
        print("  [0] Animate Full Sequence (All Successful Stages)")
        print("\n--- Or select a single trajectory to play: ---")
        for i, stage in enumerate(available_stages):
            print(
                f"  [{i + 1}] {stage['stage_name']} (Mode: {stage['execution_mode']})"
            )
        print("  [m] Back to Trial Selection")
        print("  [q] Quit")

        try:
            choice_str = (
                input(f"Enter choice (1-{len(available_stages)} or m/q): ")
                .strip()
                .lower()
            )
            if choice_str == "q":
                return "quit"
            if choice_str == "m":
                return "menu"
            if choice_str == "0":
                animate_full_trial_sequence(
                    pin_viz, model, data, trial_df, args, stage_order
                )
                continue  # Go back to the menu after animation

            selected_stage_data = available_stages[int(choice_str) - 1]

            # Find the previous stage to set the robot's static starting pose
            try:
                current_stage_index = stage_order.index(
                    selected_stage_data["stage_name"]
                )
                if current_stage_index > 0:
                    for i in range(current_stage_index - 1, -1, -1):
                        prev_stage_name = stage_order[i]
                        if (
                            prev_stage_name in stage_data_map
                            and stage_data_map[prev_stage_name]["traj_jaco"]
                        ):
                            prev_traj_jaco = stage_data_map[prev_stage_name][
                                "traj_jaco"
                            ]
                            if prev_traj_jaco and prev_traj_jaco.get("points"):
                                last_waypoint = prev_traj_jaco["points"][-1]
                                selected_stage_data["static_jaco_config"] = {
                                    "positions": last_waypoint["positions"],
                                    "joint_names": prev_traj_jaco["joint_names"],
                                }
                                break
            except ValueError:
                pass

            # Enter the playback loop for the chosen trajectory
            action = trajectory_playback_loop(
                pin_viz, model, data, selected_stage_data, args
            )
            if action == "quit":
                return "quit"
            # If 'back', the loop will continue, re-displaying this menu

        except (ValueError, IndexError):
            print("Invalid choice.")


def animate_full_trial_sequence(pin_viz, model, data, trial_df, args, stage_order):
    """
    Gathers all successful trajectories for a trial, sorts them, and plays
    them back as a single continuous animation.
    """
    print("\n--- Preparing Full Trial Animation ---")

    # 1. Filter for successful stages and sort them chronologically
    successful_stages = trial_df[trial_df["status"] == "Success"].to_dict("records")
    if not successful_stages:
        print("No successful stages to animate.")
        input("Press Enter to continue...")
        return

    def get_stage_index(stage):
        try:
            return stage_order.index(stage["stage_name"])
        except ValueError:
            return float("inf")

    successful_stages.sort(key=get_stage_index)

    # 2. Build the master list of joint configurations (q) for the full animation
    q_sequence = []
    q_current = pin.neutral(model)  # Start from the neutral pose

    print("Stitching trajectories:")
    for stage in successful_stages:
        print(f"  - Processing stage: {stage['stage_name']}")
        traj_jaco = stage.get("traj_jaco")
        traj_atool = stage.get("traj_atool")
        execution_mode = stage.get("execution_mode")

        # Get joint maps for the current stage's trajectories
        jaco_map = get_joint_map(model, (traj_jaco or {}).get("joint_names", []))
        atool_map = get_joint_map(model, (traj_atool or {}).get("joint_names", []))

        # Determine waypoints for this stage
        jaco_points = (traj_jaco or {}).get("points", [])
        atool_points = (traj_atool or {}).get("points", [])

        if execution_mode == "Sequential":
            num_waypoints = len(jaco_points) + len(atool_points)
        else:
            num_waypoints = max(len(jaco_points), len(atool_points))

        if num_waypoints == 0:
            continue

        for i in range(num_waypoints):
            q_frame = q_current.copy()  # Start from the previous stage's end state

            if execution_mode == "Sequential":
                if i < len(jaco_points):  # Jaco movement phase
                    update_q_from_waypoint(
                        q_frame,
                        model,
                        jaco_points[i],
                        traj_jaco["joint_names"],
                        jaco_map,
                    )
                else:  # Articutool movement phase
                    # Hold Jaco at its final position
                    update_q_from_waypoint(
                        q_frame,
                        model,
                        jaco_points[-1],
                        traj_jaco["joint_names"],
                        jaco_map,
                    )
                    # Move Articutool
                    atool_idx = i - len(jaco_points)
                    update_q_from_waypoint(
                        q_frame,
                        model,
                        atool_points[atool_idx],
                        traj_atool["joint_names"],
                        atool_map,
                    )
            else:  # Synchronous, Jaco Only, Articutool Only
                if jaco_points:
                    jaco_idx = min(i, len(jaco_points) - 1)
                    update_q_from_waypoint(
                        q_frame,
                        model,
                        jaco_points[jaco_idx],
                        traj_jaco["joint_names"],
                        jaco_map,
                    )
                if atool_points:
                    atool_idx = min(i, len(atool_points) - 1)
                    update_q_from_waypoint(
                        q_frame,
                        model,
                        atool_points[atool_idx],
                        traj_atool["joint_names"],
                        atool_map,
                    )

            q_sequence.append(q_frame)

        # Carry over the final state of this stage for the next one
        if q_sequence:
            q_current = q_sequence[-1]

    # 3. Animate the full sequence
    if not q_sequence:
        print("No animation frames were generated.")
        input("Press Enter to continue...")
        return

    print(
        f"\nAnimating {len(successful_stages)} stages ({len(q_sequence)} frames)... Press Ctrl+C to stop."
    )
    try:
        # Set the visualizer to the first frame before starting
        pin_viz.display(q_sequence[0])
        time.sleep(1)

        for i, q in enumerate(q_sequence):
            pin_viz.display(q)
            print(f"  Displaying frame {i + 1}/{len(q_sequence)}", end="\r")
            time.sleep(1.0 / args.fps)
        print("\nAnimation finished.")
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")
    input("Press Enter to return to the menu...")


def main(args):
    """Main execution function with the new trial-centric workflow."""
    print("--- Interactive Benchmark Trajectory Visualizer ---")

    df = load_benchmark_data(args.benchmark_files)
    if df.empty:
        print("No data could be loaded. Exiting.")
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
        trial_df = select_trial(df)
        if trial_df is None or trial_df.empty:
            break

        action = trial_visualization_loop(pin_viz, model, data, trial_df, args)
        if action == "quit":
            break

    print("Visualizer finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize robot trajectories from benchmark files."
    )
    parser.add_argument(
        "benchmark_files", nargs="+", help="One or more benchmark JSON result files."
    )
    parser.add_argument(
        "--xacro_file",
        type=str,
        default="/home/regulus/ada_ws/src/ada_ros2/ada_moveit/config/ada.urdf.xacro",
        help="Path to the robot URDF/XACRO file.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for animation."
    )
    args = parser.parse_args()
    main(args)
