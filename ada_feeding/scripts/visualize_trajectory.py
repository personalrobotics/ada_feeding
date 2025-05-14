#!/usr/bin/env python3

import pinocchio as pin
import pinocchio.visualize  # Ensure this is imported for MeshcatVisualizer
import meshcat
import numpy as np
import json
import subprocess
import tempfile
import os
import argparse
import time
import sys

from typing import Optional, Dict, Any, List  # Added List

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
        # Attempt to find ros2 executable, assuming it's in PATH
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
            ]  # Use os.pathsep

        # Add common workspace paths if specific package_dirs are not found or to supplement
        # This is a heuristic and might need adjustment based on your workspace structure
        common_workspace_paths = [
            os.path.expanduser("~/ros2_ws/src"),
            os.path.expanduser("~/workspace/src"),
            # Add other common workspace locations if necessary
        ]
        for wsp in common_workspace_paths:
            if os.path.isdir(wsp) and wsp not in package_dirs:
                # Pinocchio expects a list of directories containing packages, not the package directories themselves
                # So, if wsp is like '/home/user/ros2_ws/src', it's correct.
                package_dirs.append(wsp)

        # If the current working directory's parent might be a workspace 'src'
        # This helps if running from within a package.
        current_ws_src_path = os.path.abspath(
            os.path.join(os.getcwd(), "..", "..", "src")
        )
        if (
            os.path.isdir(current_ws_src_path)
            and current_ws_src_path not in package_dirs
        ):
            package_dirs.append(current_ws_src_path)

        if package_dirs:
            logger_func(
                f"Using package_dirs for Pinocchio: {list(set(package_dirs))}"
            )  # Show unique dirs
            model = pin.buildModelFromUrdf(
                temp_urdf_path, package_dirs=list(set(package_dirs))
            )
            collision_model = pin.buildGeomFromUrdf(
                model,
                temp_urdf_path,
                pin.GeometryType.COLLISION,
                package_dirs=list(set(package_dirs)),
            )
            visual_model = pin.buildGeomFromUrdf(
                model,
                temp_urdf_path,
                pin.GeometryType.VISUAL,
                package_dirs=list(set(package_dirs)),
            )
        else:
            logger_func(
                "Warning: ROS_PACKAGE_PATH not found or empty. Mesh loading might fail if using package:// paths."
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
            # logger_func(f"Temporary URDF file for debugging: {temp_urdf_path}") # Uncomment to keep file
            os.remove(temp_urdf_path)


def load_enhanced_trajectory_from_json(
    filepath: str, logger_func=print
) -> Optional[Dict[str, Any]]:
    """Loads the enhanced trajectory data from a JSON file."""
    try:
        with open(filepath, "r") as f:
            traj_data = json.load(f)
        logger_func(f"Enhanced trajectory loaded from {filepath}")
        # Basic validation for the new structure
        if "jaco_joint_names" not in traj_data or "waypoints" not in traj_data:
            logger_func(
                "Error: Loaded JSON is missing 'jaco_joint_names' or 'waypoints'."
            )
            return None
        return traj_data
    except FileNotFoundError:
        logger_func(f"Error: Trajectory file not found at {filepath}")
    except json.JSONDecodeError:
        logger_func(f"Error: Could not decode JSON from {filepath}")
    except Exception as e:
        logger_func(f"An unexpected error occurred loading trajectory: {e}")
    return None


def get_pinocchio_joint_info(
    model: pin.Model, joint_name: str
) -> Optional[Dict[str, Any]]:
    """Gets q_idx_start, nq, and nv for a named joint in the Pinocchio model."""
    if model.existJointName(joint_name):
        joint_id = model.getJointId(joint_name)
        joint_obj = model.joints[joint_id]
        if joint_obj.nq > 0:  # Only consider actuated joints
            return {
                "name": joint_name,
                "q_idx_start": joint_obj.idx_q,
                "nq": joint_obj.nq,
                "nv": joint_obj.nv,
                "id": joint_id,
            }
    return None


def main(xacro_file: str, trajectory_file: str):
    print("--- Pinocchio + MeshCat Trajectory Visualizer (Enhanced) ---")

    urdf_string = xacro_to_urdf_string(xacro_file)
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
        for i in range(1, model.njoints):  # Start from 1 to skip universe
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
        visualizer = (
            meshcat.Visualizer().open()
        )  # open=True is default if not passed to initViewer
        # visualizer.wait() # Wait for browser to connect
        print(f"MeshCat viewer URL: {visualizer.url()}")
        pin_viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
        pin_viz.initViewer(viewer=visualizer)  # Already opened
        pin_viz.loadViewerModel(
            rootNodeName=model.name if model.name else "pinocchio_robot"
        )
        print("MeshCat viewer initialized and model loaded.")
    except Exception as e:
        print(f"Error initializing MeshCat or Pinocchio visualizer: {e}")
        return

    trajectory_data = load_enhanced_trajectory_from_json(trajectory_file)
    if not trajectory_data:
        print("Failed to load valid trajectory data. Exiting.")
        return

    jaco_joint_names_from_traj = trajectory_data["jaco_joint_names"]
    waypoints_data = trajectory_data["waypoints"]
    if not waypoints_data:
        print("Trajectory contains no waypoints. Exiting.")
        return

    q = pin.neutral(model)
    print(
        f"Neutral configuration q (size {model.nq}): {q.T if model.nq > 0 else 'N/A'}"
    )

    # Map Jaco joint names from trajectory to Pinocchio model indices
    jaco_joint_mappings: List[Optional[Dict[str, Any]]] = []
    for name_in_traj in jaco_joint_names_from_traj:
        jaco_joint_mappings.append(get_pinocchio_joint_info(model, name_in_traj))

    if any(m is None for m in jaco_joint_mappings):
        print(
            "Warning: Some Jaco joints from trajectory not found or not actuated in model:"
        )
        for i, name in enumerate(jaco_joint_names_from_traj):
            if jaco_joint_mappings[i] is None:
                print(f"  - {name}")

    # Get Pinocchio info for Articutool joints
    articutool_pitch_joint_info = get_pinocchio_joint_info(
        model, ARTICUTOOL_PITCH_JOINT_NAME
    )
    articutool_roll_joint_info = get_pinocchio_joint_info(
        model, ARTICUTOOL_ROLL_JOINT_NAME
    )

    if not articutool_pitch_joint_info:
        print(
            f"Error: Articutool pitch joint '{ARTICUTOOL_PITCH_JOINT_NAME}' not found or not actuated in model. Exiting."
        )
        return
    if not articutool_roll_joint_info:
        print(
            f"Error: Articutool roll joint '{ARTICUTOOL_ROLL_JOINT_NAME}' not found or not actuated in model. Exiting."
        )
        return

    print("\nArticutool Joint Mappings:")
    print(
        f"  Pitch ('{ARTICUTOOL_PITCH_JOINT_NAME}'): Maps to q[{articutool_pitch_joint_info['q_idx_start']}], nq={articutool_pitch_joint_info['nq']}"
    )
    print(
        f"  Roll  ('{ARTICUTOOL_ROLL_JOINT_NAME}'): Maps to q[{articutool_roll_joint_info['q_idx_start']}], nq={articutool_roll_joint_info['nq']}"
    )

    def set_q_from_waypoint(q_vector: np.ndarray, waypoint: Dict[str, Any]):
        """Updates q_vector with Jaco and Articutool positions from a waypoint."""
        # Set Jaco joints
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
                elif nq == 2 and nv == 1:  # Continuous (cos/sin)
                    q_vector[q_idx_start] = np.cos(theta_traj)
                    q_vector[q_idx_start + 1] = np.sin(theta_traj)

        # Set Articutool joints
        if waypoint.get("articutool_waypoint_feasible", False):
            pitch_sol = waypoint.get("articutool_pitch_solution_rad")
            roll_sol = waypoint.get("articutool_roll_solution_rad")

            if pitch_sol is not None and articutool_pitch_joint_info:
                q_idx = articutool_pitch_joint_info["q_idx_start"]
                # Assuming nq=1 for these revolute joints
                if articutool_pitch_joint_info["nq"] == 1:
                    q_vector[q_idx] = pitch_sol
                elif (
                    articutool_pitch_joint_info["nq"] == 2
                    and articutool_pitch_joint_info["nv"] == 1
                ):  # cos/sin
                    q_vector[q_idx] = np.cos(pitch_sol)
                    q_vector[q_idx + 1] = np.sin(pitch_sol)

            if roll_sol is not None and articutool_roll_joint_info:
                q_idx = articutool_roll_joint_info["q_idx_start"]
                if articutool_roll_joint_info["nq"] == 1:
                    q_vector[q_idx] = roll_sol
                elif (
                    articutool_roll_joint_info["nq"] == 2
                    and articutool_roll_joint_info["nv"] == 1
                ):  # cos/sin
                    q_vector[q_idx] = np.cos(roll_sol)
                    q_vector[q_idx + 1] = np.sin(roll_sol)
        else:
            # Articutool not feasible at this waypoint.
            # Optionally, set Articutool joints to a default/neutral pose (e.g., 0.0)
            # This provides a visual cue. If not set, they retain previous values.
            if articutool_pitch_joint_info:
                q_idx = articutool_pitch_joint_info["q_idx_start"]
                if articutool_pitch_joint_info["nq"] == 1:
                    q_vector[q_idx] = 0.0
                elif (
                    articutool_pitch_joint_info["nq"] == 2
                    and articutool_pitch_joint_info["nv"] == 1
                ):
                    q_vector[q_idx] = np.cos(0.0)
                    q_vector[q_idx + 1] = np.sin(0.0)
            if articutool_roll_joint_info:
                q_idx = articutool_roll_joint_info["q_idx_start"]
                if articutool_roll_joint_info["nq"] == 1:
                    q_vector[q_idx] = 0.0
                elif (
                    articutool_roll_joint_info["nq"] == 2
                    and articutool_roll_joint_info["nv"] == 1
                ):
                    q_vector[q_idx] = np.cos(0.0)
                    q_vector[q_idx + 1] = np.sin(0.0)
            # print(f"  Articutool not feasible at this waypoint. Setting its joints to 0.")

    # Initial display
    current_q_display = q.copy()  # Start with neutral
    if waypoints_data:
        set_q_from_waypoint(current_q_display, waypoints_data[0])
    pin_viz.display(current_q_display)
    q[:] = current_q_display[:]  # Update main q

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
                        set_q_from_waypoint(anim_q, waypoints_data[current_point_idx])
                        pin_viz.display(anim_q)
                        print(
                            f"  Displaying point {current_point_idx + 1}/{num_trajectory_points}",
                            end="\r",
                            flush=True,
                        )
                        time.sleep(0.05)  # Animation speed
                    print("\nAnimation finished.                                ")
                except KeyboardInterrupt:
                    print("\nAnimation stopped.                                 ")
                new_q_to_display[:] = anim_q[:]
                q[:] = new_q_to_display[:]
                pin_viz.display(q)
                continue
            elif user_input:
                print("Invalid command.")
                continue
            else:
                continue

            set_q_from_waypoint(new_q_to_display, waypoints_data[current_point_idx])
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
        description="Visualize enhanced JointTrajectory using Pinocchio and MeshCat."
    )
    parser.add_argument(
        "xacro_file",
        type=str,
        help="Path to the robot XACRO file (that instantiates the robot).",
    )
    parser.add_argument(
        "trajectory_file", type=str, help="Path to the .json ENHANCED trajectory file."
    )
    args = parser.parse_args()
    main(args.xacro_file, args.trajectory_file)
