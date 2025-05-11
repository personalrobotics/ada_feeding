#!/usr/bin/env python3

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

from typing import Optional


def xacro_to_urdf_string(xacro_filename: str, logger_func=print) -> Optional[str]:
    """Converts a Xacro file to a URDF XML string using ros2 run xacro."""
    if not os.path.exists(xacro_filename):
        logger_func(f"Error: Xacro file not found at {xacro_filename}")
        return None

    logger_func(f"Processing Xacro file: {xacro_filename}")
    try:
        ros_distro = os.environ.get("ROS_DISTRO", "your_ros2_distro")
        process = subprocess.run(
            ["ros2", "run", "xacro", "xacro", xacro_filename],
            check=True,
            capture_output=True,
            text=True,
        )
        urdf_xml_string = process.stdout
        logger_func("XACRO processing successful.")
        return urdf_xml_string
    except FileNotFoundError as e:
        logger_func(f"Fatal: Command 'ros2 run xacro ...' failed. Is xacro installed ")
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
    temp_urdf_path = ""  # Initialize to ensure it's defined in finally
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        ) as temp_file:
            temp_urdf_path = temp_file.name
            temp_file.write(urdf_xml_string)

        logger_func(f"Generated temporary URDF file: {temp_urdf_path}")

        # Attempt to guess package directories if ROS_PACKAGE_PATH is set
        package_dirs = []
        ros_package_path = os.environ.get("ROS_PACKAGE_PATH")
        if ros_package_path:
            package_dirs = [p for p in ros_package_path.split(":") if os.path.isdir(p)]

        if package_dirs:
            logger_func(f"Using package_dirs for Pinocchio: {package_dirs}")
            model = pin.buildModelFromUrdf(temp_urdf_path, package_dirs=package_dirs)
            collision_model = pin.buildGeomFromUrdf(
                model,
                temp_urdf_path,
                pin.GeometryType.COLLISION,
                package_dirs=package_dirs,
            )
            visual_model = pin.buildGeomFromUrdf(
                model,
                temp_urdf_path,
                pin.GeometryType.VISUAL,
                package_dirs=package_dirs,
            )
        else:
            logger_func(
                "Warning: ROS_PACKAGE_PATH not found or empty. Mesh loading might fail if using package:// paths without it."
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
            # For debugging URDF, comment out the next line
            # logger_func(f"Temporary URDF file for debugging: {temp_urdf_path}")
            os.remove(temp_urdf_path)
            # logger_func(f"Cleaned up temporary URDF file: {temp_urdf_path}")


def load_trajectory_from_json(filepath: str, logger_func=print):
    try:
        with open(filepath, "r") as f:
            traj_data = json.load(f)
        logger_func(f"Trajectory loaded from {filepath}")
        return traj_data
    except FileNotFoundError:
        logger_func(f"Error: Trajectory file not found at {filepath}")
    except json.JSONDecodeError:
        logger_func(f"Error: Could not decode JSON from {filepath}")
    except Exception as e:
        logger_func(f"An unexpected error occurred loading trajectory: {e}")
    return None


def main(xacro_file: str, trajectory_file: str):
    print("--- Pinocchio + MeshCat Trajectory Visualizer ---")

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
        print("\n--- Pinocchio Model Joint Details (from model.joints) ---")
        for i in range(model.njoints):
            joint_obj = model.joints[i]
            joint_name_in_model = model.names[i]
            print(
                f"IdxInModel {i}: Name='{joint_name_in_model}', PinocchioShortName='{joint_obj.shortname()}', "
                f"idx_q={joint_obj.idx_q}, nq={joint_obj.nq}, "
                f"idx_v={joint_obj.idx_v}, nv={joint_obj.nv}, "
                f"idInModel={joint_obj.id}"
            )
            type_name = type(joint_obj).__name__
            # More specific checks for common types that might have nq > 1
            if isinstance(joint_obj, pin.JointModelFreeFlyer):
                type_name = "JointModelFreeFlyer (nq=7,nv=6)"
            elif isinstance(joint_obj, pin.JointModelSphericalZYX):
                type_name = "JointModelSphericalZYX (nq=3,nv=3)"
            elif isinstance(joint_obj, pin.JointModelSpherical):
                type_name = "JointModelSpherical (nq=4,nv=3)"
            elif isinstance(joint_obj, pin.JointModelPlanar):
                type_name = "JointModelPlanar (nq=3,nv=3)"
            elif isinstance(joint_obj, pin.JointModelTranslation):
                type_name = "JointModelTranslation (nq=3,nv=3)"
            # Check for unaligned revolute joints (often used for continuous)
            elif (
                "Rx" in type_name
                or "Ry" in type_name
                or "Rz" in type_name
                or "RevoluteUnaligned" in type_name
            ):
                if joint_obj.nq == 2 and joint_obj.nv == 1:
                    type_name += " (Likely Continuous cos/sin)"
            print(f"  Type: {type_name}")
        print("-------------------------------------------------------\n")

    print("Initializing MeshCat viewer... Waiting for connection.")
    try:
        visualizer = meshcat.Visualizer()
        print(f"MeshCat viewer URL: {visualizer.url()}")
        pin_viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
        pin_viz.initViewer(viewer=visualizer, open=True)
        pin_viz.loadViewerModel(
            rootNodeName=model.name if model.name else "pinocchio_robot"
        )
        print("MeshCat viewer initialized and model loaded.")
    except Exception as e:
        print(f"Error initializing MeshCat or Pinocchio visualizer: {e}")
        print(
            "Please ensure MeshCat is installed and accessible (e.g., `pip install meshcat`)."
        )
        return

    trajectory_data = load_trajectory_from_json(trajectory_file)
    if (
        not trajectory_data
        or "joint_names" not in trajectory_data
        or "points" not in trajectory_data
    ):
        print("Failed to load valid trajectory data. Exiting.")
        return

    traj_joint_names = trajectory_data["joint_names"]
    traj_points = trajectory_data["points"]
    if not traj_points:
        print("Trajectory contains no points. Exiting.")
        return

    q = pin.neutral(model)
    print(
        f"Neutral configuration q (size {model.nq}): {q.T if model.nq > 0 else 'N/A (nq=0)'}"
    )

    mapped_joint_info_for_trajectory = []
    missing_joints_in_model = []

    for name_in_traj in traj_joint_names:
        if model.existJointName(name_in_traj):
            joint_id = model.getJointId(name_in_traj)
            joint_obj = model.joints[joint_id]
            if joint_obj.nq > 0:
                mapped_joint_info_for_trajectory.append(
                    {
                        "name": name_in_traj,
                        "q_idx_start": joint_obj.idx_q,
                        "nq": joint_obj.nq,
                        "nv": joint_obj.nv,
                    }
                )
            else:
                print(
                    f"Info: Joint '{name_in_traj}' from trajectory is type '{joint_obj.shortname()}' (nq=0) in model. Will not be actuated."
                )
                mapped_joint_info_for_trajectory.append(None)
        else:
            missing_joints_in_model.append(name_in_traj)
            mapped_joint_info_for_trajectory.append(None)

    if missing_joints_in_model:
        print(
            f"Warning: Trajectory joints NOT found in Pinocchio model: {missing_joints_in_model}"
        )

    valid_mappings = [
        info for info in mapped_joint_info_for_trajectory if info is not None
    ]
    if not valid_mappings:
        print(
            "Error: None of the trajectory joints could be mapped to actuated joints in Pinocchio model. Exiting."
        )
        return

    print("\nMapped info for trajectory joints:")
    for i, info in enumerate(mapped_joint_info_for_trajectory):
        if info:
            print(
                f"  Traj Joint {i} ('{info['name']}'): Maps to q[{info['q_idx_start']}-"
                f"{info['q_idx_start'] + info['nq'] - 1}], ModelJoint_nq={info['nq']}, ModelJoint_nv={info['nv']}"
            )
        else:
            print(
                f"  Traj Joint {i} ('{traj_joint_names[i]}'): Not mapped or not actuated."
            )

    # Initial display
    current_q_display = q.copy()
    if traj_points:
        initial_positions = traj_points[0]["positions"]
        for k, mapping_info in enumerate(mapped_joint_info_for_trajectory):
            if mapping_info and k < len(initial_positions):
                theta_traj = initial_positions[k]
                q_idx_start = mapping_info["q_idx_start"]
                joint_nq = mapping_info["nq"]
                joint_nv = mapping_info["nv"]
                if joint_nq == 1:  # Standard revolute/prismatic (includes nv=0 or nv=1)
                    current_q_display[q_idx_start] = theta_traj
                elif joint_nq == 2 and joint_nv == 1:  # Continuous (cos/sin)
                    current_q_display[q_idx_start] = np.cos(theta_traj)
                    current_q_display[q_idx_start + 1] = np.sin(theta_traj)
    pin_viz.display(current_q_display)
    q[:] = current_q_display[:]  # Update main q

    current_point_idx = 0
    num_trajectory_points = len(traj_points)
    print("\n--- Trajectory Control ---")
    print("Open the MeshCat URL in your browser.")

    running = True
    while running:
        print(f"\nPoint: {current_point_idx + 1}/{num_trajectory_points}")
        print("Commands: [n]ext, [p]rev, [f]irst, [l]ast, [g <num>], [a]nimate, [q]uit")

        try:
            user_input = input("Enter command: ").strip().lower()
            new_q_to_display = (
                q.copy()
            )  # Start with current q, only update relevant parts

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
                        positions_to_set = traj_points[current_point_idx]["positions"]
                        for k, mapping_info in enumerate(
                            mapped_joint_info_for_trajectory
                        ):
                            if mapping_info and k < len(positions_to_set):
                                theta_traj = positions_to_set[k]
                                q_idx_start = mapping_info["q_idx_start"]
                                joint_nq = mapping_info["nq"]
                                joint_nv = mapping_info["nv"]
                                if joint_nq == 1:
                                    anim_q[q_idx_start] = theta_traj
                                elif joint_nq == 2 and joint_nv == 1:
                                    anim_q[q_idx_start] = np.cos(theta_traj)
                                    anim_q[q_idx_start + 1] = np.sin(theta_traj)
                        pin_viz.display(anim_q)
                        print(
                            f"  Displaying point {current_point_idx + 1}/{num_trajectory_points}",
                            end="\r",
                            flush=True,
                        )
                        time.sleep(0.05)
                    print("\nAnimation finished.                                ")
                except KeyboardInterrupt:
                    print("\nAnimation stopped.                                 ")
                new_q_to_display[:] = anim_q[:]  # Update with last animated state
                q[:] = new_q_to_display[:]
                pin_viz.display(q)  # Ensure final state is displayed once
                continue  # Skip re-applying single point after animation
            elif user_input:
                print("Invalid command.")
                continue
            else:
                continue  # Empty input, re-prompt

            # Update new_q_to_display for the current_point_idx
            if traj_points and 0 <= current_point_idx < num_trajectory_points:
                positions_to_set = traj_points[current_point_idx]["positions"]
                for k, mapping_info in enumerate(mapped_joint_info_for_trajectory):
                    if mapping_info and k < len(positions_to_set):
                        theta_traj = positions_to_set[k]
                        q_idx_start = mapping_info["q_idx_start"]
                        joint_nq = mapping_info["nq"]
                        joint_nv = mapping_info["nv"]
                        if joint_nq == 1:
                            new_q_to_display[q_idx_start] = theta_traj
                        elif joint_nq == 2 and joint_nv == 1:
                            new_q_to_display[q_idx_start] = np.cos(theta_traj)
                            new_q_to_display[q_idx_start + 1] = np.sin(theta_traj)
                        # else: # Other nq/nv cases are not set

            pin_viz.display(new_q_to_display)
            q[:] = new_q_to_display[:]  # Persist the displayed q

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
        description="Visualize saved JointTrajectory using Pinocchio and MeshCat."
    )
    parser.add_argument(
        "xacro_file",
        type=str,
        help="Path to the robot XACRO file (that instantiates the robot).",
    )
    parser.add_argument(
        "trajectory_file", type=str, help="Path to the .json trajectory file."
    )
    args = parser.parse_args()
    main(args.xacro_file, args.trajectory_file)
