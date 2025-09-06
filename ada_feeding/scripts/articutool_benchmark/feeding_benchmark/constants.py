# Standard imports
import math
import rclpy
import numpy as np

# --- Constants ---
LOGGER = rclpy.logging.get_logger("end_to_end_benchmark")

# Planning Groups
PLANNING_GROUP_JACO = "jaco_arm"
PLANNING_GROUP_ATOOL = "articutool"
PLANNING_GROUP_FULL = "jaco_arm_with_articutool"

# Joint Names
JOINT_NAMES_JACO = [f"j2n6s200_joint_{i + 1}" for i in range(6)]
JOINT_NAMES_ATOOL = [f"atool_joint{i + 1}" for i in range(2)]
JOINT_NAMES_FULL = JOINT_NAMES_JACO + JOINT_NAMES_ATOOL

# Link Names
BASE_LINK_JACO = "j2n6s200_link_base"
BASE_LINK_ATOOL = "atool_link_base"
BASE_LINK_FULL = BASE_LINK_JACO
END_EFFECTOR_LINK_JACO = "j2n6s200_end_effector"
END_EFFECTOR_LINK_ATOOL = "tool_tip"
END_EFFECTOR_LINK_FULL = END_EFFECTOR_LINK_ATOOL

# Physical & Mathematical Constants
EPSILON = 1e-6
ARTICUTOOL_PITCH_LIMITS_RAD = (-math.pi / 2, math.pi / 2)
ARTICUTOOL_ROLL_LIMITS_RAD = (-math.pi, math.pi)
WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])
ARTICUTOOL_LENGTH_M = 0.14
ARTICUTOOL_MAX_VELOCITY_RAD_S = 4.0  # Conservative limit for Dynamixel XC430

# Benchmark-Specific Parameters
PATH_CONSTRAINT_QUAT_XYZW = (0.707, 0.0, 0.0, 0.707)
PATH_CONSTRAINT_TOLERANCE_XYZ_RAD = (math.pi / 2, 2 * math.pi, math.pi / 4)
BASELINE_PATH_CONSTRAINT_TOLERANCE_XYZ_RAD = (
    np.deg2rad(10.0),
    2 * math.pi,
    np.deg2rad(10.0),
)
