# Standard imports
import time
import math
from enum import Enum
from threading import Lock
from typing import Optional, List, Dict, Tuple

# Third-party imports
from pymoveit2 import MoveIt2
import rclpy
from rclpy.node import Node
import tf2_ros

# ROS 2 message imports
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import (
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    JointConstraint,
)
from shape_msgs.msg import SolidPrimitive

# Local application imports
from .data_structures import TrialStatus
from .constants import (
    LOGGER,
    PLANNING_GROUP_JACO,
    PLANNING_GROUP_ATOOL,
    PLANNING_GROUP_FULL,
)


class MoveIt2ConstraintType(Enum):
    JOINT = "joint"
    POSITION = "position"
    ORIENTATION = "orientation"
    POSE = "pose"


def create_pose_constraint(
    pose: Pose, tolerance_position: float = 0.001, tolerance_orientation: float = 0.001
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates a standard pose goal constraint."""
    return (
        MoveIt2ConstraintType.POSE,
        {
            "pose": pose,
            "tolerance_position": tolerance_position,
            "tolerance_orientation": tolerance_orientation,
        },
    )


def create_orientation_path_constraint(
    quat_xyzw: Tuple, tolerance_rad: Tuple
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates an orientation path constraint, useful for keeping the tool level."""
    return (
        MoveIt2ConstraintType.ORIENTATION,
        {
            "quat_xyzw": Quaternion(
                x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2], w=quat_xyzw[3]
            ),
            "tolerance": tolerance_rad,
            "weight": 1.0,
        },
    )


def create_joint_constraint(
    joint_positions: List[float],
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates a joint goal constraint."""
    return (MoveIt2ConstraintType.JOINT, {"joint_positions": joint_positions})


def create_position_constraint(
    position: Point, tolerance_position: float = 0.001
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates a standard position goal constraint."""
    return (
        MoveIt2ConstraintType.POSITION,
        {
            "position": position,
            "tolerance": tolerance_position,
            "weight": 1.0,
        },
    )


class MotionPlanner:
    """A wrapper for MoveIt2 to provide a seamless, synchronous API for planning."""

    def __init__(
        self,
        node: Node,
        moveit2_jaco: MoveIt2,
        moveit2_atool: Optional[MoveIt2],
        moveit2_full: Optional[MoveIt2],
        tf_buffer: tf2_ros.Buffer,
        planning_timeout: float,
    ):
        self._node = node
        self._planning_timeout = planning_timeout
        self._tf_buffer = tf_buffer
        self._moveit2_objects = {
            PLANNING_GROUP_JACO: moveit2_jaco,
            PLANNING_GROUP_ATOOL: moveit2_atool,
            PLANNING_GROUP_FULL: moveit2_full,
        }
        self._lock = Lock()

    def _get_planner(self, group_name: str) -> MoveIt2:
        if group_name not in self._moveit2_objects:
            raise ValueError(f"Unknown planning group: {group_name}")
        return self._moveit2_objects[group_name]

    def _create_orientation_constraint_msg(
        self, planner: MoveIt2, kwargs: Dict
    ) -> OrientationConstraint:
        constraint = OrientationConstraint()
        constraint.header.frame_id = planner.base_link_name
        constraint.link_name = kwargs.get("target_link", planner.end_effector_name)
        constraint.orientation = kwargs["quat_xyzw"]
        tolerance = kwargs.get("tolerance", 0.001)
        tolerance_xyz = (
            (tolerance, tolerance, tolerance)
            if isinstance(tolerance, float)
            else tolerance
        )
        constraint.absolute_x_axis_tolerance = tolerance_xyz[0]
        constraint.absolute_y_axis_tolerance = tolerance_xyz[1]
        constraint.absolute_z_axis_tolerance = tolerance_xyz[2]
        constraint.weight = kwargs.get("weight", 1.0)
        return constraint

    def _create_position_constraint_msg(
        self, planner: MoveIt2, kwargs: Dict
    ) -> PositionConstraint:
        constraint = PositionConstraint()
        constraint.header.frame_id = planner.base_link_name
        constraint.link_name = kwargs.get("target_link", planner.end_effector_name)
        pose = Pose()
        pose.position = kwargs["position"]
        constraint.constraint_region.primitive_poses.append(pose)
        tolerance = kwargs.get("tolerance", 0.001)
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[tolerance])
        constraint.constraint_region.primitives.append(sphere)
        constraint.weight = kwargs.get("weight", 1.0)
        return constraint

    def _create_joint_constraint_msg(
        self, planner: MoveIt2, kwargs: Dict
    ) -> List[JointConstraint]:
        joint_constraints = []
        if "joint_names" not in kwargs or "joint_positions" not in kwargs:
            self._node.get_logger().error(
                "Joint constraints require 'joint_names' and 'joint_positions'."
            )
            return joint_constraints
        for name, pos in zip(kwargs["joint_names"], kwargs["joint_positions"]):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = pos
            constraint.tolerance_above = kwargs.get("tolerance", 0.01)
            constraint.tolerance_below = kwargs.get("tolerance", 0.01)
            constraint.weight = kwargs.get("weight", 1.0)
            joint_constraints.append(constraint)
        return joint_constraints

    @staticmethod
    def _scale_cartesian_trajectory_velocity(
        traj: JointTrajectory, scale_factor: float
    ):
        for point in traj.points:
            nsec = (point.time_from_start.sec * 1e9) + point.time_from_start.nanosec
            nsec /= scale_factor
            sec = int(math.floor(nsec / 1e9))
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = int(nsec - (sec * 1e9))
            for i in range(len(point.velocities)):
                point.velocities[i] *= scale_factor
            for i in range(len(point.accelerations)):
                point.accelerations[i] *= scale_factor**2

    def _transform_goal_to_base_link(
        self, planner: MoveIt2, constraint: Tuple[MoveIt2ConstraintType, Dict]
    ):
        constraint_type, kwargs = constraint
        if (
            constraint_type != MoveIt2ConstraintType.POSE
            or "frame_id" not in kwargs
            or kwargs["frame_id"] is None
        ):
            return
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = kwargs["frame_id"]
        pose_stamped.pose = kwargs["pose"]
        transformed_pose = self._tf_buffer.transform(
            pose_stamped, planner.base_link_name
        )
        kwargs["pose"] = transformed_pose.pose
        kwargs["frame_id"] = planner.base_link_name

    def compute_ik(
        self,
        group_name: str,
        target_pose: Pose,
        start_joint_state: Optional[List[float]] = None,
    ) -> Optional[JointState]:
        planner = self._get_planner(group_name)
        if planner is None:
            LOGGER.error(f"Cannot compute IK. Planner for '{group_name}' unavailable.")
            return None
        with self._lock:
            ik_solution = planner.compute_ik(
                position=target_pose.position,
                quat_xyzw=target_pose.orientation,
                start_joint_state=start_joint_state,
            )
        return ik_solution

    def compute_constrained_ik(
        self,
        group_name: str,
        target_pose: Pose,
        start_joint_state: List[float],
        ik_constraints: List[Tuple[MoveIt2ConstraintType, Dict]],
    ) -> Optional[JointState]:
        planner = self._get_planner(group_name)
        if planner is None:
            LOGGER.error(f"Cannot compute IK. Planner for '{group_name}' unavailable.")
            return None
        constraints_msg = Constraints()
        for constraint_type, kwargs in ik_constraints:
            if constraint_type == MoveIt2ConstraintType.ORIENTATION:
                msg = self._create_orientation_constraint_msg(planner, kwargs)
                constraints_msg.orientation_constraints.append(msg)
            elif constraint_type == MoveIt2ConstraintType.POSITION:
                msg = self._create_position_constraint_msg(planner, kwargs)
                constraints_msg.position_constraints.append(msg)
            elif constraint_type == MoveIt2ConstraintType.JOINT:
                msgs = self._create_joint_constraint_msg(planner, kwargs)
                constraints_msg.joint_constraints.extend(msgs)
        with self._lock:
            ik_solution = planner.compute_ik(
                position=target_pose.position,
                quat_xyzw=target_pose.orientation,
                start_joint_state=start_joint_state,
                constraints=constraints_msg,
            )
        return ik_solution

    def compute_fk(
        self, group_name: str, joint_state: JointState, fk_link_names: List[str]
    ) -> Optional[List[PoseStamped]]:
        planner = self._get_planner(group_name)
        if planner is None:
            LOGGER.error(f"Cannot compute FK. Planner for '{group_name}' unavailable.")
            return None
        with self._lock:
            fk_poses = planner.compute_fk(
                joint_state=joint_state, fk_link_names=fk_link_names
            )
        return fk_poses

    def plan(
        self,
        group_name: str,
        start_state: Optional[List[float]],
        goal_constraints: List[Tuple[MoveIt2ConstraintType, Dict]],
        path_constraints: Optional[List[Tuple[MoveIt2ConstraintType, Dict]]] = None,
        planning_time: float = 5.0,
        target_link: Optional[str] = None,
        cartesian: bool = False,
        cartesian_max_step: float = 0.001,
        cartesian_jump_threshold: float = 5.0,
        cartesian_fraction_threshold: float = 0.92,
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        planner = self._get_planner(group_name)
        if planner is None:
            LOGGER.error(
                f"Cannot compute plan. Planner for '{group_name}' unavailable."
            )
            return TrialStatus.PLANNER_FAILURE, None, 0.0
        if not goal_constraints:
            LOGGER.error("Planning failed: At least one goal constraint is required.")
            return TrialStatus.IK_FAILURE, None, 0.0
        if cartesian:
            planner.cartesian_jump_threshold = cartesian_jump_threshold
        with self._lock:
            planner.clear_goal_constraints()
            planner.clear_path_constraints()
            if cartesian:
                try:
                    for constraint in goal_constraints:
                        self._transform_goal_to_base_link(planner, constraint)
                except Exception as e:
                    LOGGER.error(
                        f"Failed to transform Cartesian goal to base link: {e}"
                    )
                    return TrialStatus.IK_FAILURE, None, 0.0
            for constraint_type, kwargs in goal_constraints:
                if constraint_type == MoveIt2ConstraintType.JOINT:
                    planner.set_joint_goal(**kwargs)
                elif constraint_type == MoveIt2ConstraintType.POSITION:
                    planner.set_position_goal(**kwargs, target_link=target_link)
                elif constraint_type == MoveIt2ConstraintType.ORIENTATION:
                    planner.set_orientation_goal(**kwargs, target_link=target_link)
                elif constraint_type == MoveIt2ConstraintType.POSE:
                    planner.set_pose_goal(**kwargs, target_link=target_link)
            planner.allowed_planning_time = planning_time
            planner.cartesian_avoid_collisions = True
            if path_constraints:
                for constraint_type, kwargs in path_constraints:
                    if constraint_type == MoveIt2ConstraintType.JOINT:
                        planner.set_path_joint_constraint(**kwargs)
                    elif constraint_type == MoveIt2ConstraintType.POSITION:
                        planner.set_path_position_constraint(**kwargs)
                    elif constraint_type == MoveIt2ConstraintType.ORIENTATION:
                        planner.set_path_orientation_constraint(**kwargs)
            future = planner.plan_async(
                start_joint_state=start_state,
                cartesian=cartesian,
                max_step=cartesian_max_step,
            )
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start_time > self._planning_timeout:
                future.cancel()
                return TrialStatus.PLANNER_FAILURE, None, self._planning_timeout
            time.sleep(0.1)
        planning_time = time.time() - start_time
        traj = planner.get_trajectory(
            future,
            cartesian=cartesian,
            cartesian_fraction_threshold=cartesian_fraction_threshold,
        )
        if not traj or not traj.points:
            return TrialStatus.PLANNER_FAILURE, None, planning_time
        if cartesian and planner.max_velocity > 0.0:
            MotionPlanner._scale_cartesian_trajectory_velocity(
                traj, planner.max_velocity
            )
        return TrialStatus.SUCCESS, traj, planning_time
