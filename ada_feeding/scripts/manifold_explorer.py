#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script explores the "Leveling Feasibility Manifold" for the Jaco arm
equipped with the Articutool. It densely samples the robot's configuration
space and, for each sample, determines if the Articutool can achieve a
level orientation.

The output is a CSV file containing the sampled joint configurations and their
feasibility, which can then be used for visualization and analysis.
"""

# Standard imports
from datetime import datetime
import os
import math
import subprocess
import sys
import argparse
import csv
from typing import Optional, List, Tuple

# Third-party imports
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

# --- Constants ---
# These should be adjusted to match your robot's configuration
JOINT_NAMES = [f"j2n6s200_joint_{i + 1}" for i in range(6)]
END_EFFECTOR_LINK = "j2n6s200_end_effector"
WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])
ARTICUTOOL_PITCH_LIMITS_RAD = (-math.pi / 2, math.pi / 2)
ARTICUTOOL_ROLL_LIMITS_RAD = (-math.pi, math.pi)
EPSILON = 1e-6


class ManifoldExplorer:
    """
    Samples and analyzes the configuration space of the Jaco arm to find the
    Articutool's leveling feasibility manifold.
    """

    def __init__(self, xacro_file_path: str):
        self.xacro_file_path = xacro_file_path
        self.pinocchio_model: Optional[pin.Model] = None
        self.pinocchio_data: Optional[pin.Data] = None
        self.jaco_ee_frame_id_pin: Optional[int] = None
        self.joint_limits: List[Tuple[float, float]] = []

        self._initialize_pinocchio()
        if self.pinocchio_model:
            self.joint_limits = self._get_joint_limits()

    def _initialize_pinocchio(self):
        """Loads the robot model from a XACRO file into Pinocchio."""
        print("Initializing Pinocchio model from XACRO...")
        try:
            # Use ros2 run to convert xacro to urdf in memory
            process = subprocess.run(
                ["ros2", "run", "xacro", "xacro", self.xacro_file_path],
                check=True,
                capture_output=True,
                text=True,
            )
            urdf_xml_string = process.stdout
            self.pinocchio_model = pin.buildModelFromXML(urdf_xml_string)
            self.pinocchio_data = self.pinocchio_model.createData()
            self.jaco_ee_frame_id_pin = self.pinocchio_model.getFrameId(
                END_EFFECTOR_LINK
            )
            print("Pinocchio model loaded successfully.")
        except FileNotFoundError:
            print(
                f"Error: 'ros2' command not found. Is ROS 2 sourced?", file=sys.stderr
            )
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error running xacro: {e}", file=sys.stderr)
            print(f"Stderr: {e.stderr}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Failed to initialize Pinocchio model: {e}", file=sys.stderr)
            self.pinocchio_model = None

    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """Retrieves joint limits from the Pinocchio model."""
        if self.pinocchio_model is None:
            return [(-math.pi, math.pi)] * len(JOINT_NAMES)

        limits = []
        for name in JOINT_NAMES:
            if self.pinocchio_model.existJointName(name):
                joint_id = self.pinocchio_model.getJointId(name)
                # In Pinocchio, joint indices for position/velocity can differ
                idx_q = self.pinocchio_model.joints[joint_id].idx_q
                limits.append(
                    (
                        self.pinocchio_model.lowerPositionLimit[idx_q],
                        self.pinocchio_model.upperPositionLimit[idx_q],
                    )
                )
            else:
                print(
                    f"Warning: Joint '{name}' not found in Pinocchio model.",
                    file=sys.stderr,
                )
                limits.append((-math.pi, math.pi))  # Default limit
        return limits

    def generate_random_joint_config(self) -> List[float]:
        """Generates a random joint configuration within limits."""
        return [np.random.uniform(low, high) for low, high in self.joint_limits]

    def _solve_articutool_ik(
        self, target_vector: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Analytical IK solver for the Articutool."""
        vx, vy, vz = target_vector
        solutions = []

        # Check if arcsin argument is valid
        asin_arg = -vx
        if not (-1.0 - EPSILON <= asin_arg <= 1.0 + EPSILON):
            return []

        # Clamp the argument to be safe
        theta_r1 = math.asin(np.clip(asin_arg, -1.0, 1.0))
        # Find the second solution for roll
        theta_r2 = (math.pi - theta_r1 + math.pi) % (2 * math.pi) - math.pi

        for theta_r in list(set([theta_r1, theta_r2])):
            cos_tr = math.cos(theta_r)
            # Handle the case where cos(theta_r) is near zero
            if math.isclose(cos_tr, 0.0, abs_tol=EPSILON):
                # If cos(theta_r) is 0, then vy and vz must also be 0 for a solution
                if math.isclose(vy, 0.0, abs_tol=EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=EPSILON
                ):
                    # Pitch can be anything, but 0 is a reasonable choice
                    solutions.append(
                        (0.0, (theta_r + math.pi) % (2 * math.pi) - math.pi)
                    )
                continue

            # General solution for pitch
            theta_p = math.atan2(vz, vy)
            solutions.append(
                (
                    (theta_p + math.pi) % (2 * math.pi) - math.pi,
                    (theta_r + math.pi) % (2 * math.pi) - math.pi,
                )
            )
        return solutions

    def check_feasibility(self, jaco_joint_config: List[float]) -> bool:
        """
        Checks if the Articutool can maintain leveling at a single Jaco configuration.
        Returns True if a valid Articutool joint configuration exists, False otherwise.
        """
        if (
            self.pinocchio_model is None
            or self.pinocchio_data is None
            or self.jaco_ee_frame_id_pin is None
        ):
            return False

        q = pin.neutral(self.pinocchio_model)
        for j, name in enumerate(JOINT_NAMES):
            if self.pinocchio_model.existJointName(name):
                joint_id = self.pinocchio_model.getJointId(name)
                joint_obj = self.pinocchio_model.joints[joint_id]
                # Handle different joint types (revolute vs. prismatic, etc.)
                if joint_obj.nq == 2 and not joint_obj.shortname().startswith(
                    "JointModelRX"
                ):  # e.g., for SO(2) joints
                    q[joint_obj.idx_q : joint_obj.idx_q + 2] = [
                        math.cos(jaco_joint_config[j]),
                        math.sin(jaco_joint_config[j]),
                    ]
                else:  # Standard revolute joint
                    q[joint_obj.idx_q] = jaco_joint_config[j]

        # Run forward kinematics to get frame placements
        pin.forwardKinematics(self.pinocchio_model, self.pinocchio_data, q)
        pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)

        # Get the end-effector's transformation matrix
        ee_transform = self.pinocchio_data.oMf[self.jaco_ee_frame_id_pin]

        # Determine the "up" vector in the end-effector's frame
        target_up_in_ee_frame = (
            R.from_matrix(ee_transform.rotation).inv().apply(WORLD_UP_VECTOR)
        )

        # Solve the Articutool's IK for this target vector
        solutions = self._solve_articutool_ik(target_up_in_ee_frame)

        if not solutions:
            return False

        # Check if any solution is within the Articutool's joint limits
        return any(
            ARTICUTOOL_PITCH_LIMITS_RAD[0] <= pitch <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
            and ARTICUTOOL_ROLL_LIMITS_RAD[0] <= roll <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
            for pitch, roll in solutions
        )

    def run_exploration(self, num_samples: int, output_file: str):
        """
        Main exploration loop. Samples the C-space and saves results to a file.
        """
        if not self.pinocchio_model:
            print(
                "Cannot run exploration: Pinocchio model not loaded.", file=sys.stderr
            )
            return

        print(f"Starting exploration of {num_samples} samples...")
        results = []
        header = [f"j{i + 1}" for i in range(len(JOINT_NAMES))] + ["is_feasible"]

        for i in range(num_samples):
            config = self.generate_random_joint_config()
            is_feasible = self.check_feasibility(config)
            results.append(config + [1 if is_feasible else 0])

            if (i + 1) % 1000 == 0:
                print(f"  ...processed {i + 1}/{num_samples} samples.")

        print(f"Exploration complete. Saving results to {output_file}...")
        try:
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(results)
            print("Successfully saved results.")
        except IOError as e:
            print(f"Error saving file: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Explore the Articutool's leveling feasibility manifold."
    )
    parser.add_argument(
        "xacro_file", type=str, help="Path to the robot URDF/XACRO file."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of random configurations to sample.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="manifold_data",
        help="Directory to save the output CSV file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.xacro_file):
        print(f"Error: XACRO file not found at '{args.xacro_file}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(args.output_dir, f"manifold_samples_{timestamp}.csv")

    explorer = ManifoldExplorer(args.xacro_file)
    explorer.run_exploration(args.num_samples, output_file)


if __name__ == "__main__":
    main()
