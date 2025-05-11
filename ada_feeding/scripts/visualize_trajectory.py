#!/usr/bin/env python3

import pinocchio as pin
import pinocchio.visualize  # Import the submodule
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
        # Try to find a ROS 2 distribution to use in error messages if needed
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
    temp_urdf_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        ) as temp_file:
            temp_urdf_path = temp_file.name
            temp_file.write(urdf_xml_string)

        logger_func(f"Generated temporary URDF file: {temp_urdf_path}")

        # For Pinocchio to find meshes, especially if using package://,
        # it might need directories to search.
        # Example: model = pin.buildModelFromUrdf(temp_urdf_path, package_dirs=['/path/to/your_ros_ws/src/your_robot_description_pkg'])
        # If your meshes are relative to the URDF, and the URDF is generated from Xacro,
        # ensure the Xacro resolves paths correctly or provide absolute paths in Xacro,
        # or explore pin.loadUrdfModel with package_dirs.
        # For now, we assume meshes are findable or simple geometries are used.

        model = pin.buildModelFromUrdf(temp_urdf_path)
        # Pinocchio automatically creates collision and visual models from the URDF
        collision_model = pin.buildGeomFromUrdf(
            model, temp_urdf_path, pin.GeometryType.COLLISION
        )
        visual_model = pin.buildGeomFromUrdf(
            model, temp_urdf_path, pin.GeometryType.VISUAL
        )

        data = model.createData()
        collision_data = collision_model.createData()  # if you need collision checks
        visual_data = (
            visual_model.createData()
        )  # if you need to update visual geoms (less common for just display)

        logger_func(
            f"Pinocchio model loaded successfully. Name: {model.name}, Nq: {model.nq}, Nv: {model.nv}"
        )
        logger_func(f"Number of joints in model: {model.njoints}")  # Includes universe
        # for jname in model.names: print(f"  Joint in model: {jname}")

        return model, collision_model, visual_model, data
    except Exception as e:
        logger_func(f"Error loading Pinocchio model: {e}")
        return None, None, None, None
    finally:
        if temp_urdf_file and os.path.exists(temp_urdf_path):
            os.remove(temp_urdf_path)
            logger_func(f"Cleaned up temporary URDF file: {temp_urdf_path}")


def load_trajectory_from_json(filepath: str, logger_func=print):
    """Loads a JointTrajectory from a JSON file."""
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

    # 1. Convert Xacro to URDF String
    urdf_string = xacro_to_urdf_string(xacro_file)
    if not urdf_string:
        print("Exiting due to Xacro processing failure.")
        return

    # 2. Load Pinocchio Model
    load_result = load_pinocchio_model_from_urdf_string(urdf_string)
    if not load_result or not load_result[0]:  # Check if model is loaded
        print("Exiting due to Pinocchio model loading failure.")
        return
    model, collision_model, visual_model, data = load_result

    # 3. Initialize MeshCat Visualizer
    print("Initializing MeshCat viewer... Waiting for connection.")
    try:
        # You can specify a zmq_url if you have a specific MeshCat server running
        # visualizer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        visualizer = (
            meshcat.Visualizer()
        )  # Starts its own server or connects to existing
        print(f"MeshCat viewer URL: {visualizer.url()}")

        # Create Pinocchio's MeshCat visualizer interface
        # Note: pin.visualize.MeshcatVisualizer is the correct class
        pin_viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
        pin_viz.initViewer(
            viewer=visualizer, open=True
        )  # Pass the meshcat.Visualizer instance
        pin_viz.loadViewerModel(
            rootNodeName="pinocchio_robot"
        )  # Give it a name in the scene tree
        print("MeshCat viewer initialized and model loaded.")

    except Exception as e:
        print(f"Error initializing MeshCat or Pinocchio visualizer: {e}")
        print("Please ensure MeshCat is installed and accessible.")
        return

    # 4. Load Trajectory
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

    # 5. Map Trajectory Joint Names to Pinocchio Model's q indices
    # Pinocchio's q includes all DoFs, including the root joint if it's a free flyer.
    # For a fixed-base manipulator, model.nq usually matches the number of actuated joints.
    # For a floating base, the first 7 elements of q are [tx,ty,tz, qx,qy,qz,qw] for the base.

    # Get the neutral configuration (often zeros for fixed base, or identity for floating base part)
    q = pin.neutral(model)
    print(f"Neutral configuration q (size {model.nq}): {q.T}")

    # Create a list of indices in q that correspond to the trajectory joint names
    # and their order.
    q_indices_for_traj_joints = []
    missing_joints_in_model = []

    for name_in_traj in traj_joint_names:
        if model.existJointName(name_in_traj):
            joint_id = model.getJointId(name_in_traj)
            joint_obj = model.joints[joint_id]

            # idx_q is the starting index in q for this joint's configuration
            # nq is the number of parameters for this joint's configuration (1 for revolute/prismatic)
            if joint_obj.nq == 1:  # Assuming all trajectory joints are 1-DoF
                q_indices_for_traj_joints.append(joint_obj.idx_q)
            else:
                print(
                    f"Warning: Joint '{name_in_traj}' in model has {joint_obj.nq} DoFs. Trajectory assumes 1-DoF. Skipping this joint."
                )
                q_indices_for_traj_joints.append(None)  # Placeholder
        else:
            missing_joints_in_model.append(name_in_traj)
            q_indices_for_traj_joints.append(None)  # Placeholder

    if missing_joints_in_model:
        print(
            f"Warning: The following joints from the trajectory were NOT found in the Pinocchio model: {missing_joints_in_model}"
        )

    if not any(idx is not None for idx in q_indices_for_traj_joints):
        print(
            "Error: None of the trajectory joints could be mapped to the Pinocchio model. Exiting."
        )
        return

    print(f"Mapped q-indices for trajectory joints: {q_indices_for_traj_joints}")

    # Display initial pose (neutral or first trajectory point)
    if traj_points:
        initial_positions = traj_points[0]["positions"]
        for i, q_idx in enumerate(q_indices_for_traj_joints):
            if q_idx is not None and i < len(initial_positions):
                q[q_idx] = initial_positions[i]
    pin_viz.display(q)

    # --- Visualization Loop ---
    current_point_idx = 0
    num_trajectory_points = len(traj_points)
    print("\n--- Trajectory Control ---")
    print("Open the MeshCat URL in your browser if it didn't open automatically.")

    running = True
    while running:
        print(f"\nPoint: {current_point_idx + 1}/{num_trajectory_points}")
        print("Commands: [n]ext, [p]rev, [f]irst, [l]ast, [g <num>], [a]nimate, [q]uit")

        try:
            user_input = input("Enter command: ").strip().lower()

            if user_input == "n":
                if current_point_idx < num_trajectory_points - 1:
                    current_point_idx += 1
                else:
                    print("Already at the last point.")
            elif user_input == "p":
                if current_point_idx > 0:
                    current_point_idx -= 1
                else:
                    print("Already at the first point.")
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
                except (IndexError, ValueError):
                    print("Invalid format. Use 'g <number>'.")
            elif user_input == "a":
                print("Animating trajectory... Press Ctrl+C to stop animation.")
                start_anim_idx = current_point_idx
                try:
                    for i in range(start_anim_idx, num_trajectory_points):
                        current_point_idx = i
                        positions_to_set = traj_points[current_point_idx]["positions"]
                        temp_q = q.copy()  # Start from the full neutral/previous q
                        for k, q_idx in enumerate(q_indices_for_traj_joints):
                            if q_idx is not None and k < len(positions_to_set):
                                temp_q[q_idx] = positions_to_set[k]
                        pin_viz.display(temp_q)
                        print(
                            f"  Displaying point {current_point_idx + 1}/{num_trajectory_points}",
                            end="\r",
                            flush=True,
                        )
                        time.sleep(0.05)  # Animation speed
                    q[:] = temp_q[:]  # Update main q to last animated state
                    print("\nAnimation finished.                                ")
                except KeyboardInterrupt:
                    q[:] = temp_q[:]  # Update main q to where animation stopped
                    print("\nAnimation stopped.                                 ")
            elif user_input == "q":
                running = False
                print("Quitting.")
                break
            else:
                if user_input:
                    print("Invalid command.")

            # Update q for the current point_idx and display
            if traj_points and 0 <= current_point_idx < num_trajectory_points:
                positions_to_set = traj_points[current_point_idx]["positions"]
                # q = pin.neutral(model) # Reset q to neutral before applying current point for fixed-base
                # Or, update incrementally from previous q for smoother transitions if some joints are not in traj
                for i, q_idx in enumerate(q_indices_for_traj_joints):
                    if q_idx is not None and i < len(positions_to_set):
                        q[q_idx] = positions_to_set[i]
                pin_viz.display(q)

        except EOFError:  # Handle Ctrl+D
            print("\nEOF received, quitting.")
            running = False
        except KeyboardInterrupt:
            print("\nInterrupted, quitting.")
            running = False
        except Exception as e:
            print(f"An error occurred in the loop: {e}")
            # Decide if you want to break or continue
            # running = False

    print("Visualizer finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a saved JointTrajectory using Pinocchio and MeshCat."
    )
    parser.add_argument("xacro_file", type=str, help="Path to the robot.xacro file.")
    parser.add_argument(
        "trajectory_file", type=str, help="Path to the .json trajectory file."
    )

    args = parser.parse_args()
    main(args.xacro_file, args.trajectory_file)
