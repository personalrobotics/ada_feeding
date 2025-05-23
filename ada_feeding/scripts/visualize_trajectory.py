#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script is used to visualize robot trajectories or single joint states
using Pinocchio and MeshCat. It can load:
1. Enhanced trajectory JSON files (containing Jaco and Articutool waypoints).
2. Single joint state JSON files (representing a sensor_msgs/msg/JointState).
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
from typing import Optional, Dict, Any, List

# Define Articutool joint names as they appear in the Pinocchio model
# (Verify these names from the script's "Pinocchio Model Joint Details" printout)
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
            os.path.expanduser("~/ros2_ws/src"),  # Common Foxy/Galactic/Humble
            os.path.expanduser("~/dev_ws/src"),  # Common Iron+
            os.path.expanduser("~/workspace/src"),
        ]
        for wsp in common_workspace_paths:
            if os.path.isdir(wsp) and wsp not in package_dirs:
                package_dirs.append(wsp)

        # Attempt to find the 'src' directory of the current ROS 2 workspace
        # This is a heuristic based on typical workspace layouts.
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):  # Stop at root
            if (
                os.path.basename(current_dir) == "install"
                or os.path.basename(current_dir) == "build"
                or os.path.basename(current_dir) == "log"
            ):
                ws_root = os.path.dirname(current_dir)
                src_path = os.path.join(ws_root, "src")
                if os.path.isdir(src_path) and src_path not in package_dirs:
                    package_dirs.append(src_path)
                break
            current_dir = os.path.dirname(current_dir)

        if package_dirs:
            unique_package_dirs = list(set(package_dirs))  # Remove duplicates
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
                "Warning: ROS_PACKAGE_PATH not found or empty, and common workspace paths not found. Mesh loading might fail if using package:// paths."
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


def load_enhanced_trajectory_from_json(
    filepath: str, logger_func=print
) -> Optional[Dict[str, Any]]:
    """Loads the enhanced trajectory data from a JSON file."""
    try:
        with open(filepath, "r") as f:
            traj_data = json.load(f)
        logger_func(f"Enhanced trajectory loaded from {filepath}")
        if "jaco_joint_names" not in traj_data or "waypoints" not in traj_data:
            logger_func(
                "Error: Loaded JSON is missing 'jaco_joint_names' or 'waypoints'."
            )
            return None
        return traj_data
    except FileNotFoundError:
        logger_func(f"Error: Trajectory file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        logger_func(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        logger_func(f"An unexpected error occurred loading trajectory: {e}")
        return None


def load_joint_state_from_json(
    filepath: str, logger_func=print
) -> Optional[Dict[str, Any]]:
    """Loads joint state data (expected sensor_msgs/msg/JointState format) from a JSON file."""
    try:
        with open(filepath, "r") as f:
            joint_state_data = json.load(f)
        logger_func(f"Joint state loaded from {filepath}")
        if "name" not in joint_state_data or "position" not in joint_state_data:
            logger_func(
                "Error: Loaded JSON for joint state is missing 'name' or 'position' fields."
            )
            return None
        if len(joint_state_data["name"]) != len(joint_state_data["position"]):
            logger_func(
                "Error: Mismatch between number of names and positions in joint state JSON."
            )
            return None
        return joint_state_data
    except FileNotFoundError:
        logger_func(f"Error: Joint state file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        logger_func(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        logger_func(f"An unexpected error occurred loading joint state: {e}")
        return None


def get_pinocchio_joint_info(
    model: pin.Model, joint_name: str
) -> Optional[Dict[str, Any]]:
    """Gets q_idx_start, nq, and nv for a named joint in the Pinocchio model."""
    if model.existJointName(joint_name):
        joint_id = model.getJointId(joint_name)
        joint_obj = model.joints[joint_id]
        # Consider only joints that contribute to the configuration vector q
        if joint_obj.idx_q >= 0 and joint_obj.nq > 0:
            return {
                "name": joint_name,
                "q_idx_start": joint_obj.idx_q,
                "nq": joint_obj.nq,
                "nv": joint_obj.nv,
                "id": joint_id,
            }
    return None


def set_q_from_joint_state_data(
    q_vector: np.ndarray,
    model: pin.Model,
    joint_names: List[str],
    joint_positions: List[float],
    logger_func=print,
):
    """Updates Pinocchio q_vector based on names and positions from a JointState-like structure."""
    if len(joint_names) != len(joint_positions):
        logger_func(
            "Error in set_q_from_joint_state_data: name and position lists have different lengths."
        )
        return

    name_to_pos = {name: joint_positions[i] for i, name in enumerate(joint_names)}

    for i in range(
        1, model.njoints
    ):  # Iterate through Pinocchio model joints (skip universe)
        joint_name_in_model = model.names[i]
        joint_info = get_pinocchio_joint_info(model, joint_name_in_model)

        if joint_info and joint_name_in_model in name_to_pos:
            theta_input = name_to_pos[joint_name_in_model]
            q_idx_start, nq, nv = (
                joint_info["q_idx_start"],
                joint_info["nq"],
                joint_info["nv"],
            )

            if nq == 1:  # Typically for prismatic or revolute if not using cos/sin
                q_vector[q_idx_start] = theta_input
            elif nq == 2 and nv == 1:  # Revolute joint with cos/sin representation
                q_vector[q_idx_start] = np.cos(theta_input)
                q_vector[q_idx_start + 1] = np.sin(theta_input)
            # Add other joint types if necessary (e.g., free flyer)
            # else: logger_func(f"Joint {joint_name_in_model} has nq={nq}, nv={nv} - unhandled for direct q setting from JointState.")
        # else: logger_func(f"Joint {joint_name_in_model} from model not in provided joint state or not actuated.")


def main(args):
    print("--- Pinocchio + MeshCat Visualizer ---")

    urdf_string = xacro_to_urdf_string(args.xacro_file)
    if not urdf_string:
        print("Exiting due to Xacro processing failure.")
        return

    load_result = load_pinocchio_model_from_urdf_string(urdf_string)
    if not load_result or not load_result[0]:
        print("Exiting due to Pinocchio model loading failure.")
        return
    model, collision_model, visual_model, data = load_result

    if model:
        print("\n--- Pinocchio Model Joint Details ---")
        for i in range(1, model.njoints):
            joint_name_in_model = model.names[i]
            joint_obj = model.joints[i]
            print(
                f"IdxInModel {i}: Name='{joint_name_in_model}', PinocchioType='{joint_obj.shortname()}', "
                f"idx_q={joint_obj.idx_q if joint_obj.idx_q >= 0 else 'N/A'}, nq={joint_obj.nq}, "
                f"idx_v={joint_obj.idx_v if joint_obj.idx_v >= 0 else 'N/A'}, nv={joint_obj.nv}"
            )
        print("-----------------------------------\n")

    print("Initializing MeshCat viewer... Waiting for connection.")
    try:
        visualizer = meshcat.Visualizer().open()
        print(f"MeshCat viewer URL: {visualizer.url()}")
        pin_viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
        pin_viz.initViewer(viewer=visualizer)
        pin_viz.loadViewerModel(
            rootNodeName=model.name if model.name else "pinocchio_robot"
        )
        print("MeshCat viewer initialized and model loaded.")
    except Exception as e:
        print(f"Error initializing MeshCat or Pinocchio visualizer: {e}")
        return

    q = pin.neutral(model)
    print(
        f"Neutral configuration q (size {model.nq}): {q.T if model.nq > 0 else 'N/A'}"
    )

    if args.ik_solution_json:
        print(f"\n--- Visualizing Single IK Solution from: {args.ik_solution_json} ---")
        joint_state_data = load_joint_state_from_json(args.ik_solution_json)
        if not joint_state_data:
            print("Failed to load IK solution JSON. Exiting.")
            return

        # Set q from the loaded joint state
        set_q_from_joint_state_data(
            q, model, joint_state_data["name"], joint_state_data["position"]
        )

        print(f"Displaying configuration from IK solution: {q.T}")
        pin_viz.display(q)
        try:
            input("Robot displayed. Press Enter to quit.")
        except KeyboardInterrupt:
            print("\nQuitting by user interrupt.")
        return  # Exit after displaying single state

    # --- Original Trajectory Visualization Logic ---
    if not args.trajectory_file:
        print(
            "No trajectory file or IK solution JSON provided. Displaying neutral pose."
        )
        pin_viz.display(q)
        try:
            input("Neutral pose displayed. Press Enter to quit.")
        except KeyboardInterrupt:
            print("\nQuitting by user interrupt.")
        return

    trajectory_data = load_enhanced_trajectory_from_json(args.trajectory_file)
    if not trajectory_data:
        print("Failed to load valid trajectory data. Exiting.")
        return

    jaco_joint_names_from_traj = trajectory_data["jaco_joint_names"]
    waypoints_data = trajectory_data["waypoints"]
    if not waypoints_data:
        print("Trajectory contains no waypoints. Exiting.")
        return

    jaco_joint_mappings: List[Optional[Dict[str, Any]]] = [
        get_pinocchio_joint_info(model, name) for name in jaco_joint_names_from_traj
    ]
    articutool_pitch_joint_info = get_pinocchio_joint_info(
        model, ARTICUTOOL_PITCH_JOINT_NAME
    )
    articutool_roll_joint_info = get_pinocchio_joint_info(
        model, ARTICUTOOL_ROLL_JOINT_NAME
    )

    if not articutool_pitch_joint_info or not articutool_roll_joint_info:
        print("Error: Articutool pitch or roll joint not found in model. Exiting.")
        return

    print("\nArticutool Joint Mappings:")
    print(
        f"  Pitch ('{ARTICUTOOL_PITCH_JOINT_NAME}'): Maps to q[{articutool_pitch_joint_info['q_idx_start']}], nq={articutool_pitch_joint_info['nq']}"
    )
    print(
        f"  Roll  ('{ARTICUTOOL_ROLL_JOINT_NAME}'): Maps to q[{articutool_roll_joint_info['q_idx_start']}], nq={articutool_roll_joint_info['nq']}"
    )

    def set_q_from_enhanced_waypoint(q_vector: np.ndarray, waypoint: Dict[str, Any]):
        jaco_positions = waypoint.get("jaco_positions_rad", [])
        for k, mapping_info in enumerate(jaco_joint_mappings):
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

        if waypoint.get("articutool_waypoint_feasible", False):
            pitch_sol, roll_sol = (
                waypoint.get("articutool_pitch_solution_rad"),
                waypoint.get("articutool_roll_solution_rad"),
            )
            for sol, info in [
                (pitch_sol, articutool_pitch_joint_info),
                (roll_sol, articutool_roll_joint_info),
            ]:
                if sol is not None and info:
                    q_idx, nq, nv = info["q_idx_start"], info["nq"], info["nv"]
                    if nq == 1:
                        q_vector[q_idx] = sol
                    elif nq == 2 and nv == 1:
                        q_vector[q_idx], q_vector[q_idx + 1] = np.cos(sol), np.sin(sol)
        else:  # Articutool not feasible, set to neutral (0.0)
            for info in [articutool_pitch_joint_info, articutool_roll_joint_info]:
                if info:
                    q_idx, nq, nv = info["q_idx_start"], info["nq"], info["nv"]
                    if nq == 1:
                        q_vector[q_idx] = 0.0
                    elif nq == 2 and nv == 1:
                        q_vector[q_idx], q_vector[q_idx + 1] = np.cos(0.0), np.sin(0.0)

    current_q_display = q.copy()
    if waypoints_data:
        set_q_from_enhanced_waypoint(current_q_display, waypoints_data[0])
    pin_viz.display(current_q_display)
    q[:] = current_q_display[:]

    current_point_idx = 0
    num_trajectory_points = len(waypoints_data)
    print("\n--- Trajectory Control ---")
    print("Open the MeshCat URL in your browser.")
    running = True
    while running:
        wp_info = waypoints_data[current_point_idx]
        at_feasible = wp_info.get("articutool_waypoint_feasible", False)
        at_pitch = wp_info.get("articutool_pitch_solution_rad", "N/A")
        at_roll = wp_info.get("articutool_roll_solution_rad", "N/A")
        if isinstance(at_pitch, float):
            at_pitch = f"{at_pitch:.3f}"
        if isinstance(at_roll, float):
            at_roll = f"{at_roll:.3f}"
        print(
            f"\nPoint: {current_point_idx + 1}/{num_trajectory_points} | Articutool Feasible: {at_feasible} (P: {at_pitch}, R: {at_roll})"
        )
        print("Commands: [n]ext, [p]rev, [f]irst, [l]ast, [g <num>], [a]nimate, [q]uit")
        try:
            user_input = input("Enter command: ").strip().lower()
            new_q_to_display = q.copy()
            if user_input == "q":
                running = False
                print("Quitting.")
                break
            elif user_input == "n":
                current_point_idx = min(
                    current_point_idx + 1, num_trajectory_points - 1
                )
            elif user_input == "p":
                current_point_idx = max(current_point_idx - 1, 0)
            elif user_input == "f":
                current_point_idx = 0
            elif user_input == "l":
                current_point_idx = num_trajectory_points - 1
            elif user_input.startswith("g "):
                try:
                    target_idx = int(user_input.split(" ")[1]) - 1
                    if 0 <= target_idx < num_trajectory_points:
                        current_point_idx = target_idx
                    else:
                        print(f"Point number out of range (1-{num_trajectory_points}).")
                        continue
                except:
                    print("Invalid format. Use 'g <number>'.")
                    continue
            elif user_input == "a":
                print("Animating trajectory... Press Ctrl+C to stop animation.")
                start_anim_idx = current_point_idx
                anim_q = q.copy()
                try:
                    for i in range(start_anim_idx, num_trajectory_points):
                        current_point_idx = i
                        set_q_from_enhanced_waypoint(
                            anim_q, waypoints_data[current_point_idx]
                        )
                        pin_viz.display(anim_q)
                        print(
                            f"  Displaying point {current_point_idx + 1}/{num_trajectory_points}",
                            end="\r",
                            flush=True,
                        )
                        time.sleep(0.05)
                    print(
                        "\nAnimation finished.                                       "
                    )
                except KeyboardInterrupt:
                    print(
                        "\nAnimation stopped.                                          "
                    )
                new_q_to_display[:] = anim_q[:]
                q[:] = new_q_to_display[:]
                pin_viz.display(q)
                continue
            elif user_input:
                print("Invalid command.")
                continue
            else:
                continue
            set_q_from_enhanced_waypoint(
                new_q_to_display, waypoints_data[current_point_idx]
            )
            pin_viz.display(new_q_to_display)
            q[:] = new_q_to_display[:]
        except EOFError:
            print("\nEOF received, quitting.")
            running = False
        except KeyboardInterrupt:
            print("\nInterrupted, quitting.")
            running = False
        except Exception as e:
            print(f"An error occurred in the loop: {e}")
    print("Visualizer finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize robot trajectories or single IK solutions using Pinocchio and MeshCat."
    )
    parser.add_argument("xacro_file", type=str, help="Path to the robot XACRO file.")

    group = parser.add_mutually_exclusive_group(
        required=False
    )  # Make providing one of these optional
    group.add_argument(
        "--trajectory_file",
        type=str,
        help="Path to the .json ENHANCED trajectory file.",
    )
    group.add_argument(
        "--ik_solution_json",
        type=str,
        help="Path to a .json file representing a single sensor_msgs/msg/JointState for IK visualization.",
    )

    args = parser.parse_args()

    if not args.trajectory_file and not args.ik_solution_json:
        print(
            "Neither --trajectory_file nor --ik_solution_json provided. Will display neutral pose."
        )
        # main will handle displaying neutral if both are None after model load.

    main(args)
