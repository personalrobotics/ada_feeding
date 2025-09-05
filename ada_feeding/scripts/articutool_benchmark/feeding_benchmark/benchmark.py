# Standard imports
import os
import time
import math
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

# Third-party imports
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import tf2_ros
from pymoveit2 import MoveIt2

# ROS 2 message imports
from geometry_msgs.msg import Pose, Quaternion, Point
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.node import Node

# Local application imports
from .constants import (
    LOGGER,
    BASE_LINK_JACO,
    JOINT_NAMES_JACO,
    JOINT_NAMES_ATOOL,
    JOINT_NAMES_FULL,
    PLANNING_GROUP_JACO,
    PLANNING_GROUP_ATOOL,
    PLANNING_GROUP_FULL,
    END_EFFECTOR_LINK_JACO,
    END_EFFECTOR_LINK_ATOOL,
    END_EFFECTOR_LINK_FULL,
    ARTICUTOOL_PITCH_LIMITS_RAD,
    ARTICUTOOL_ROLL_LIMITS_RAD,
    PATH_CONSTRAINT_QUAT_XYZW,
    PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
    BASELINE_PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
)
from .data_structures import (
    TrialStatus,
    ExecutionMode,
    SceneGenerationParams,
    asdict,
)
from .motion_planner import (
    MotionPlanner,
    create_pose_constraint,
    create_orientation_path_constraint,
    create_joint_constraint,
    create_position_constraint,
)
from .kinematics import PinocchioModel
from .scene_generator import SceneGenerator
from . import trajectory_utils
from . import kinematic_solvers
from . import metrics
from . import results


class EndToEndBenchmark:
    """Manages the end-to-end benchmark planning process."""

    def __init__(
        self,
        node: Node,
        moveit2_jaco: MoveIt2,
        moveit2_atool: Optional[MoveIt2],
        moveit2_full: Optional[MoveIt2],
        xacro_file_path: str,
        num_trials: int = 100,
        planning_timeout: float = 5.0,
        output_dir: Optional[str] = None,
        mode: str = "articutool",
    ):
        self.node = node
        self.num_trials = num_trials
        self.mode = mode

        # --- Mode-Aware Configuration ---
        if self.mode == "6dof_baseline":
            self.jaco_ee_link = "forkTip"
        else:
            self.jaco_ee_link = END_EFFECTOR_LINK_JACO

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Use the .jsonl extension for JSON Lines format
            self.output_filename = os.path.join(
                output_dir, f"benchmark_{self.mode}.jsonl"
            )
            # Log a message confirming where the JSONL file will be saved.
            # Use the original ROS 2 LOGGER.
            LOGGER.info(f"Benchmark results will be saved to: {self.output_filename}")

        self.motion_planner = MotionPlanner(
            node,
            moveit2_jaco,
            moveit2_atool,
            moveit2_full,
            tf2_ros.Buffer(),
            planning_timeout,
        )
        # Pinocchio model for feasibility checks
        self.kinematics_model = PinocchioModel(xacro_file_path)

    def _extract_ordered_joint_solution(
        self, ik_solution: JointState
    ) -> Optional[List[float]]:
        """
        Extracts the 8 actuated joint values from a raw IK solution message
        and returns them in the correct [jaco, atool] order.
        """
        if not ik_solution or not ik_solution.name:
            return None

        # Create a dictionary mapping joint names to their positions
        solution_map = dict(zip(ik_solution.name, ik_solution.position))

        # Build the new, correctly ordered 8-element list
        ordered_solution = []
        all_joint_names = JOINT_NAMES_JACO + JOINT_NAMES_ATOOL

        for name in all_joint_names:
            if name not in solution_map:
                LOGGER.error(f"Required joint '{name}' not found in IK solution.")
                return None
            ordered_solution.append(solution_map[name])

        return ordered_solution

    # --- Planning Primitive ---
    def _plan_to_above_plate(
        self, above_plate_pose: Pose, start_state_jaco: Any
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        LOGGER.info("  Planning to AbovePlate pose...")
        goal_constraints = [
            create_pose_constraint(
                above_plate_pose, tolerance_position=0.1, tolerance_orientation=0.1
            )
        ]

        return self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=goal_constraints,
            target_link=END_EFFECTOR_LINK_JACO,
        )

    def _plan_to_level_articutool(
        self,
        jaco_wrist_pose: Pose,
        start_state_atool: List[float],
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        """
        Calculates the required joint angles for the Articutool to achieve a level
        pose and plans a trajectory to that configuration.
        """
        LOGGER.info("  Calculating Articutool leveling configuration...")

        # 1. Compute the target joint angles for leveling based on the wrist's pose
        target_leveling_joints = kinematic_solvers.compute_leveling_joints(
            jaco_wrist_pose
        )

        if target_leveling_joints is None:
            LOGGER.warning("  Could not find an IK solution for Articutool leveling.")
            return TrialStatus.IK_FAILURE, None, 0.0

        # 2. Create a joint goal constraint for the Articutool
        goal_constraints = [create_joint_constraint(target_leveling_joints)]

        # 3. Plan a joint-space motion for the Articutool group
        status, trajectory, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_ATOOL,
            start_state=start_state_atool,
            goal_constraints=goal_constraints,
        )

        if status != TrialStatus.SUCCESS:
            LOGGER.warning("  Articutool leveling plan failed.")
            return TrialStatus.PLANNER_FAILURE, None, planning_time

        LOGGER.info("  Articutool leveling plan successful.")
        return TrialStatus.SUCCESS, trajectory, planning_time

    def _plan_to_resting(
        self,
        resting_wrist_pose: Pose,
        start_state_jaco: List[float],
    ) -> Tuple[
        TrialStatus,
        Optional[JointTrajectory],
        Optional[JointTrajectory],
        Dict[str, Any],
        float,
    ]:
        """
        Plans a 6-DOF guided motion for the Jaco arm to a resting wrist pose
        and measures its leveling feasibility.
        """
        LOGGER.info("  Planning to Resting pose (S2-Heuristic)...")

        # 1. Get the current pose of the Jaco EE via Forward Kinematics
        start_joint_state = JointState(name=JOINT_NAMES_JACO, position=start_state_jaco)
        fk_poses = self.motion_planner.compute_fk(
            group_name=PLANNING_GROUP_JACO,
            joint_state=start_joint_state,
            fk_link_names=[END_EFFECTOR_LINK_JACO],
        )
        if not fk_poses:
            LOGGER.error("  FK failed, cannot generate dynamic path constraint.")
            return TrialStatus.IK_FAILURE, None, None, 0.0, 0.0

        current_ee_pose = fk_poses[0].pose

        # 2. Extract the current yaw from the wrist's orientation
        q_current = current_ee_pose.orientation
        R_current = R.from_quat([q_current.x, q_current.y, q_current.z, q_current.w])
        z_axis_ee = R_current.apply([0.0, 0.0, 1.0])
        current_yaw_rad = math.atan2(z_axis_ee[1], z_axis_ee[0])

        # --- REVISED: Construct the dynamic target orientation directly ---
        # This is the robust way to create the orientation we need.

        # Define the desired axes of our target frame in world coordinates:
        # Y-axis (up) should align with the world's Z-axis.
        y_axis_target = np.array([0.0, 0.0, 1.0])

        # Z-axis (forward) should be horizontal and point along the current yaw.
        z_axis_target = np.array(
            [math.cos(current_yaw_rad), math.sin(current_yaw_rad), 0.0]
        )

        # X-axis (left) is the cross product to form a right-handed system.
        x_axis_target = np.cross(y_axis_target, z_axis_target)

        # Create the rotation matrix from these axes and convert to a quaternion.
        # The axes vectors form the columns of the rotation matrix.
        rotation_matrix = np.array([x_axis_target, y_axis_target, z_axis_target]).T
        R_dynamic_target = R.from_matrix(rotation_matrix)
        q_dynamic_target = R_dynamic_target.as_quat()
        # --- End of Revision ---

        # 4. Define goal and the new dynamic path constraints
        goal_constraints = [
            create_position_constraint(
                resting_wrist_pose.position, tolerance_position=0.1
            ),
            create_orientation_path_constraint(
                quat_xyzw=(
                    q_dynamic_target[0],
                    q_dynamic_target[1],
                    q_dynamic_target[2],
                    q_dynamic_target[3],
                ),
                tolerance_rad=(0.1, 2 * np.pi, 0.1),
            ),
        ]
        path_constraints = [
            create_orientation_path_constraint(
                quat_xyzw=(
                    q_dynamic_target[0],
                    q_dynamic_target[1],
                    q_dynamic_target[2],
                    q_dynamic_target[3],
                ),
                tolerance_rad=PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
            )
        ]
        # Analyze the start pose against the (potentially corrected) path constraint
        LOGGER.info("  Analyzing start pose against final path constraint...")
        error_report = metrics.compute_moveit_orientation_error(
            current_ee_pose, path_constraints[0]
        )
        details = error_report["details"]
        LOGGER.info(f"    - Constraint Satisfied: {error_report['is_satisfied']}")
        for axis, data in details.items():
            err_deg = math.degrees(data["error_rad"])
            tol_deg = math.degrees(data["tolerance_rad"])
            status = "OK" if data["satisfied"] else "FAILED"
            LOGGER.info(
                f"    - {axis.title():<10}: Error = {err_deg:6.1f}°, Tolerance = ±{tol_deg:.1f}° -> {status}"
            )
        # 2. Plan the guided 6-DOF trajectory for the Jaco arm
        status, traj_jaco, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=goal_constraints,
            path_constraints=path_constraints,
            planning_time=20.0,
        )
        if status != TrialStatus.SUCCESS:
            return TrialStatus.PLANNER_FAILURE, None, None, {}, planning_time

        # 3. VERIFY the Jaco trajectory for leveling feasibility
        verification_results = metrics.verify_trajectory(
            traj_jaco, self.kinematics_model
        )
        if not verification_results.get(
            "is_kinematically_feasible", False
        ) or not verification_results.get("is_dynamically_feasible", False):
            LOGGER.warning(
                f"  Path to Resting failed verification. "
                f"Kinematic: {verification_results.get('is_kinematically_feasible', False)} "
                f"({verification_results.get('feasible_percent', 0.0):.1f}%). "
                f"Dynamic: {verification_results.get('is_dynamically_feasible', False)} "
                f"(Max required vel: {verification_results.get('max_required_velocity_rad_s', 0.0):.2f} rad/s)."
            )
            return (
                TrialStatus.VERIFICATION_FAILURE,
                traj_jaco,
                None,
                verification_results,  # Return the full dict for logging
                planning_time,
            )

        # 4. Generate the corresponding synchronous Articutool trajectory
        LOGGER.info("  Generating synchronous Articutool trajectory...")
        traj_atool = trajectory_utils.generate_leveling_atool_trajectory(
            traj_jaco, self.kinematics_model
        )

        if traj_atool is None:
            LOGGER.error("  Failed to generate synchronous Articutool trajectory.")
            return (
                TrialStatus.IK_FAILURE,
                traj_jaco,
                None,
                verification_results,
                planning_time,
            )
        return (
            TrialStatus.SUCCESS,
            traj_jaco,
            traj_atool,
            verification_results,
            planning_time,
        )

    def _plan_to_staging(
        self,
        staging_wrist_pose: Pose,
        start_state_jaco: List[float],
    ) -> Tuple[
        TrialStatus,
        Optional[JointTrajectory],
        Optional[JointTrajectory],
        Dict[str, Any],
        float,
    ]:
        """
        Plans a 6-DOF guided motion for the Jaco arm to a staging wrist pose
        and measures its leveling feasibility.
        """
        LOGGER.info("  Planning to Staging pose (S2-Heuristic)...")

        # 1. Get the start pose's orientation (q_start)
        start_joint_state = JointState(name=JOINT_NAMES_JACO, position=start_state_jaco)
        fk_poses = self.motion_planner.compute_fk(
            group_name=PLANNING_GROUP_JACO,
            joint_state=start_joint_state,
            fk_link_names=[END_EFFECTOR_LINK_JACO],
        )
        if not fk_poses:
            LOGGER.error("  FK failed, cannot generate dynamic path constraint.")
            return TrialStatus.IK_FAILURE, None, None, {}, 0.0

        current_ee_pose = fk_poses[0].pose
        q_start = R.from_quat(
            [
                current_ee_pose.orientation.x,
                current_ee_pose.orientation.y,
                current_ee_pose.orientation.z,
                current_ee_pose.orientation.w,
            ]
        )

        # 2. Get the goal pose's orientation (q_goal)
        q_goal = R.from_quat(
            [
                staging_wrist_pose.orientation.x,
                staging_wrist_pose.orientation.y,
                staging_wrist_pose.orientation.z,
                staging_wrist_pose.orientation.w,
            ]
        )

        # 3. Use SLERP to find the midpoint orientation for the path constraint
        # Ensure canonical quaternions (short path) for correct interpolation
        if np.dot(q_start.as_quat(), q_goal.as_quat()) < 0:
            q_goal_negated_array = -q_goal.as_quat()
            q_goal = R.from_quat(q_goal_negated_array)

        key_rots = R.from_quat([q_start.as_quat(), q_goal.as_quat()])
        slerp = Slerp([0, 1], key_rots)
        q_midpoint = slerp(0.5).as_quat()  # Get the orientation at t=0.5

        # 4. Build the path constraint using the midpoint orientation
        path_constraints = [
            create_orientation_path_constraint(
                quat_xyzw=(
                    q_midpoint[0],
                    q_midpoint[1],
                    q_midpoint[2],
                    q_midpoint[3],
                ),
                tolerance_rad=PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
            )
        ]
        goal_constraints = [
            create_pose_constraint(staging_wrist_pose, tolerance_position=0.1)
        ]

        LOGGER.info("  Analyzing start pose against final path constraint...")
        error_report = metrics.compute_moveit_orientation_error(
            current_ee_pose, path_constraints[0]
        )
        details = error_report["details"]
        LOGGER.info(f"    - Constraint Satisfied: {error_report['is_satisfied']}")
        for axis, data in details.items():
            err_deg = math.degrees(data["error_rad"])
            tol_deg = math.degrees(data["tolerance_rad"])
            status = "OK" if data["satisfied"] else "FAILED"
            LOGGER.info(
                f"    - {axis.title():<10}: Error = {err_deg:6.1f}°, Tolerance = ±{tol_deg:.1f}° -> {status}"
            )
        LOGGER.info("  Analyzing end pose against final path constraint...")
        error_report = metrics.compute_moveit_orientation_error(
            staging_wrist_pose, path_constraints[0]
        )
        details = error_report["details"]
        LOGGER.info(f"    - Constraint Satisfied: {error_report['is_satisfied']}")
        for axis, data in details.items():
            err_deg = math.degrees(data["error_rad"])
            tol_deg = math.degrees(data["tolerance_rad"])
            status = "OK" if data["satisfied"] else "FAILED"
            LOGGER.info(
                f"    - {axis.title():<10}: Error = {err_deg:6.1f}°, Tolerance = ±{tol_deg:.1f}° -> {status}"
            )

        status, traj_jaco, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=goal_constraints,
            path_constraints=path_constraints,
            planning_time=20.0,
        )
        if status != TrialStatus.SUCCESS:
            return TrialStatus.PLANNER_FAILURE, None, None, {}, planning_time

        verification_results = metrics.verify_trajectory(
            traj_jaco, self.kinematics_model
        )
        if not verification_results.get(
            "is_kinematically_feasible", False
        ) or not verification_results.get("is_dynamically_feasible", False):
            LOGGER.warning(
                f"  Path to Staging failed verification. "
                f"Kinematic: {verification_results.get('is_kinematically_feasible', False)} "
                f"({verification_results.get('feasible_percent', 0.0):.1f}%). "
                f"Dynamic: {verification_results.get('is_dynamically_feasible', False)} "
                f"(Max required vel: {verification_results.get('max_required_velocity_rad_s', 0.0):.2f} rad/s)."
            )
            return (
                TrialStatus.VERIFICATION_FAILURE,
                traj_jaco,
                None,
                verification_results,
                planning_time,
            )

        traj_atool = trajectory_utils.generate_leveling_atool_trajectory(
            traj_jaco, self.kinematics_model
        )
        if traj_atool is None:
            return (
                TrialStatus.IK_FAILURE,
                traj_jaco,
                None,
                verification_results,
                planning_time,
            )

        return (
            TrialStatus.SUCCESS,
            traj_jaco,
            traj_atool,
            verification_results,
            planning_time,
        )

    def _plan_presentation_from_staging(
        self,
        presentation_pose: Pose,
        start_state_jaco: List[float],
    ) -> Tuple[
        TrialStatus, Optional[JointTrajectory], Optional[JointTrajectory], float
    ]:
        """
        Plans a direct Cartesian motion from Staging to Presentation by calculating
        a geometric offset, avoiding the need for a full IK solve.
        """
        LOGGER.info("  Planning direct Cartesian motion to Presentation pose...")

        # 1. Get the starting pose of the Jaco EE via Forward Kinematics.
        start_joint_state = JointState(name=JOINT_NAMES_JACO, position=start_state_jaco)
        fk_poses = self.motion_planner.compute_fk(
            group_name=PLANNING_GROUP_JACO,
            joint_state=start_joint_state,
            fk_link_names=[END_EFFECTOR_LINK_JACO],
        )
        if not fk_poses:
            LOGGER.info("  FK failed for starting EE pose, cannot plan presentation.")
            return TrialStatus.IK_FAILURE, None, None, 0.0
        start_ee_pose = fk_poses[0].pose

        # 2. Calculate the static transform from the EE to the tool tip.
        #    This assumes the Articutool is at its zero configuration ([0,0]) to stay level.
        T_ee_tip = self.kinematics_model.get_relative_transform(
            parent_frame=END_EFFECTOR_LINK_JACO,
            child_frame=END_EFFECTOR_LINK_ATOOL,
            jaco_joints=start_state_jaco,
            atool_joints=[0.0, 0.0],
        )
        if T_ee_tip is None:
            LOGGER.info("  Could not calculate tool tip offset.")
            return TrialStatus.IK_FAILURE, None, None, 0.0

        # 3. Calculate the target Jaco EE pose.
        #    The orientation should remain the same as the staging pose.
        target_ee_orientation = start_ee_pose.orientation
        R_ee = R.from_quat(
            [
                target_ee_orientation.x,
                target_ee_orientation.y,
                target_ee_orientation.z,
                target_ee_orientation.w,
            ]
        )

        #    The target position is the tool tip's goal minus the rotated offset.
        p_tip_target = np.array(
            [
                presentation_pose.position.x,
                presentation_pose.position.y,
                presentation_pose.position.z,
            ]
        )
        t_offset_world = R_ee.apply(T_ee_tip.translation)
        p_ee_target = p_tip_target - t_offset_world

        target_ee_pose = Pose(
            position=Point(x=p_ee_target[0], y=p_ee_target[1], z=p_ee_target[2]),
            orientation=target_ee_orientation,
        )

        # 4. Plan the Cartesian motion for the Jaco arm.
        goal_constraints = [create_pose_constraint(target_ee_pose)]
        status, traj_jaco, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=goal_constraints,
            cartesian=True,
        )

        if status != TrialStatus.SUCCESS:
            return status, None, None, planning_time

        # 5. Create a holding trajectory for the Articutool.
        traj_atool = trajectory_utils.generate_hold_trajectory(
            traj_jaco, JOINT_NAMES_ATOOL, [0.0, 0.0]
        )

        return TrialStatus.SUCCESS, traj_jaco, traj_atool, planning_time

    def _plan_level_and_extract_baseline(
        self,
        start_state_jaco: List[float],
        in_food_pose: Pose,
        extraction_height_m: float = 0.05,
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        """
        Plans a single, synchronous "level-and-extract" Cartesian motion for
        the 6-DOF baseline.

        The motion is a straight-line trajectory in 6D space from the
        `InFood` pose to a new goal that is offset vertically (world +Z)
        and has a level orientation. This represents the baseline's best
        effort at a dexterous, human-like extraction maneuver.
        """
        LOGGER.info("  Planning 'level-and-extract' maneuver (6-DOF Baseline)...")

        # 1. The start pose for the Cartesian motion is the provided `in_food_pose`.
        start_pose = in_food_pose

        # 2. Define the goal pose for the maneuver.
        goal_pose = Pose()
        # The position is offset vertically in the world frame from the InFood pose.
        goal_pose.position.x = start_pose.position.x
        goal_pose.position.y = start_pose.position.y
        goal_pose.position.z = start_pose.position.z + extraction_height_m
        # The orientation is level with gravity (Y-up for the Jaco EE).
        goal_pose.orientation = Quaternion(
            x=PATH_CONSTRAINT_QUAT_XYZW[0],
            y=PATH_CONSTRAINT_QUAT_XYZW[1],
            z=PATH_CONSTRAINT_QUAT_XYZW[2],
            w=PATH_CONSTRAINT_QUAT_XYZW[3],
        )

        # 3. Plan a Cartesian trajectory from the start to the goal pose.
        goal_constraints = [create_pose_constraint(goal_pose, tolerance_position=0.1)]
        status, traj_jaco, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=goal_constraints,
            cartesian=True,
            target_link=self.jaco_ee_link,
        )

        return status, traj_jaco, planning_time

    def _plan_level_and_extract_8dof_baseline(
        self,
        start_state_full: List[float],
        in_food_pose: Pose,
        extraction_height_m: float = 0.05,
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        """
        Plans a single, synchronous "level-and-extract" Cartesian motion for
        the 8-DOF baseline using the full kinematic chain.
        """
        LOGGER.info("  Planning 'level-and-extract' maneuver (8-DOF Baseline)...")

        # 1. The start pose for the Cartesian motion is the provided `in_food_pose`.
        start_pose = in_food_pose

        R_in_food = R.from_quat(
            [
                start_pose.orientation.x,
                start_pose.orientation.y,
                start_pose.orientation.z,
                start_pose.orientation.w,
            ]
        )
        # Get the forward vector (local Z) of the in_food_pose
        z_axis_in_food = R_in_food.apply([0.0, 0.0, 1.0])
        # Project it onto the XY plane to find the yaw
        current_yaw_rad = math.atan2(z_axis_in_food[1], z_axis_in_food[0])

        # Construct a new "level" orientation (Y-up) with this yaw
        target_y_axis = np.array([0.0, 0.0, 1.0])  # Level component
        target_z_axis = np.array(
            [math.cos(current_yaw_rad), math.sin(current_yaw_rad), 0.0]
        )  # Yaw component
        target_x_axis = np.cross(target_y_axis, target_z_axis)

        rotation_matrix = np.array([target_x_axis, target_y_axis, target_z_axis]).T
        R_level_target = R.from_matrix(rotation_matrix)
        q_level_target = R_level_target.as_quat()
        level_orientation = Quaternion(
            x=q_level_target[0],
            y=q_level_target[1],
            z=q_level_target[2],
            w=q_level_target[3],
        )

        # 2. Define the goal pose: vertically offset with a level orientation.
        goal_pose = Pose()
        goal_pose.position.x = start_pose.position.x
        goal_pose.position.y = start_pose.position.y
        goal_pose.position.z = start_pose.position.z + extraction_height_m
        goal_pose.orientation = level_orientation

        # 3. Plan a Cartesian trajectory using the full 8-DOF planning group.
        goal_constraints = [
            create_pose_constraint(
                goal_pose, tolerance_position=0.1, tolerance_orientation=0.01
            )
        ]
        status, traj_full, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_FULL,
            start_state=start_state_full,
            goal_constraints=goal_constraints,
            cartesian=True,
            target_link=END_EFFECTOR_LINK_FULL,
        )

        return status, traj_full, planning_time

    def _validate_start_state(self, start_state: List[float], group_name: str) -> bool:
        """
        Checks if a start state is valid (e.g., not in collision) by planning
        a trivial motion from the start state to itself.

        Returns:
            True if the start state is valid, False otherwise.
        """
        LOGGER.info("  Validating start state with a 'plan-to-self' check...")
        goal_constraints = [create_joint_constraint(start_state)]

        # Use a very short timeout, as this should be an almost instant check.
        status, _, _ = self.motion_planner.plan(
            group_name=group_name,
            start_state=start_state,
            goal_constraints=goal_constraints,
            planning_time=0.1,
        )

        if status != TrialStatus.SUCCESS:
            LOGGER.warning("  Start state is invalid (likely in collision).")
            return False

        LOGGER.info("  Start state is valid.")
        return True

    def _add_ground_plane(self):
        """
        Adds a large plane to the planning scene to represent the ground by
        publishing to the /planning_scene topic.
        """
        LOGGER.info("Adding ground plane to the planning scene...")

        # Create a publisher to the planning scene topic if it doesn't exist
        if not hasattr(self, "planning_scene_publisher"):
            self.planning_scene_publisher = self.node.create_publisher(
                PlanningScene, "/planning_scene", 10
            )
            # Give the publisher a moment to connect
            time.sleep(1.0)

        # Define the ground plane as a collision object
        collision_object = CollisionObject()
        collision_object.header.frame_id = BASE_LINK_JACO
        collision_object.id = "ground_plane"

        # Define the plane as a thin box
        plane_primitive = SolidPrimitive()
        plane_primitive.type = SolidPrimitive.BOX
        plane_primitive.dimensions = [4.0, 4.0, 0.01]  # Large in X/Y, thin in Z

        # Position the plane slightly below the robot's base to avoid initial collision
        plane_pose = Pose()
        plane_pose.position.z = -0.005

        collision_object.primitives.append(plane_primitive)
        collision_object.primitive_poses.append(plane_pose)
        collision_object.operation = CollisionObject.ADD

        # Create a PlanningScene message to publish the update
        planning_scene_update = PlanningScene()
        planning_scene_update.world.collision_objects.append(collision_object)
        planning_scene_update.is_diff = True

        # Publish the scene update
        self.planning_scene_publisher.publish(planning_scene_update)
        LOGGER.info("Published ground plane to planning scene.")
        time.sleep(1.0)  # Give a moment for the scene to update

    # --- Main Benchmark Loop ---
    def run(self):
        """Main benchmark execution loop with granular metric collection."""
        self._add_ground_plane()

        for i in range(self.num_trials):
            LOGGER.info(f"--- Running Trial {i + 1}/{self.num_trials} ---")

            # 1. Create the parameter set and the generator for this trial
            generation_params = SceneGenerationParams()
            scene_generator = SceneGenerator(generation_params)

            # 2. Generate the scene and characteristics with a single, clean call
            scene, scene_characteristics = scene_generator.generate()

            params_dict = asdict(generation_params)
            params_dict["path_constraint_tolerance_xyz_rad"] = list(
                PATH_CONSTRAINT_TOLERANCE_XYZ_RAD
            )
            params_dict["articutool_pitch_limits_rad"] = list(
                ARTICUTOOL_PITCH_LIMITS_RAD
            )
            params_dict["articutool_roll_limits_rad"] = list(ARTICUTOOL_ROLL_LIMITS_RAD)

            trial_data = {
                "trial_id": i,
                "scene_poses": {
                    "food_pose": results.serialize_pose(scene["food_pose"]),
                    "mouth_pose": results.serialize_pose(scene["mouth_pose"]),
                    "above_plate_pose": results.serialize_pose(
                        scene["above_plate_pose"]
                    ),
                    "above_food_pose": results.serialize_pose(scene["above_food_pose"]),
                    "in_food_pose": results.serialize_pose(scene["in_food_pose"]),
                    "staging_pose": results.serialize_pose(scene["staging_pose"]),
                    "resting_pose": results.serialize_pose(scene["resting_pose"]),
                },
                "scene_characteristics": scene_characteristics,
                "parameters": params_dict,
                "stages": [],
                "end_to_end_success": False,  # Default to False
                "mode": self.mode,
            }

            # 3. Initialize the robot's state for the trial
            current_jaco_state = scene["home_config"]
            current_atool_state = [0.0, 0.0]
            trial_failed = False

            # --- Stage 1: Home -> AbovePlate ---
            if not trial_failed:
                LOGGER.info("Stage 1: Home -> AbovePlate")
                status, traj_jaco, planning_time = self._plan_to_above_plate(
                    scene["above_plate_pose"], current_jaco_state
                )
                cartesian_path_length = metrics.calculate_cartesian_path_length(
                    traj_jaco,
                    PLANNING_GROUP_JACO,
                    self.kinematics_model,
                    self.jaco_ee_link,
                )
                joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                trial_data["stages"].append(
                    {
                        "stage_name": "HomeToAbovePlate",
                        "target_frame": END_EFFECTOR_LINK_JACO,
                        "status": status.value,
                        "execution_mode": ExecutionMode.JACO_ONLY.value,
                        "planning_time_sec": planning_time,
                        "trajectory_path_length_m": cartesian_path_length,
                        "total_joint_travel_rad": joint_travel,
                        "custom_metrics": {},
                        "traj_jaco": results.serialize_trajectory(traj_jaco),
                        "traj_atool": None,
                    }
                )
                if status != TrialStatus.SUCCESS:
                    LOGGER.error(f"  Stage 1 failed. Skipping trial.")
                    trial_failed = True
                else:
                    current_jaco_state = list(traj_jaco.points[-1].positions)

            # --- Stage 2: Pre-acquisition ---
            jaco_ee_above_food, jaco_ee_in_food, skewer_angles = None, None, None
            if not trial_failed:
                if self.mode == "articutool":
                    LOGGER.info("Stage 2: Pre-acquisition")
                    start_time = time.time()
                    jaco_ee_above_food, jaco_ee_in_food, skewer_angles = (
                        kinematic_solvers.find_optimal_skewer_config(
                            scene["above_food_pose"],
                            scene["in_food_pose"],
                            scene_characteristics["in_food_sampled_polar_angle_rad"],
                            self.kinematics_model,
                            self.motion_planner,
                        )
                    )
                    if skewer_angles:
                        LOGGER.info(
                            "  Optimal skewer angles (deg): "
                            f"Jaco Tilt: {np.rad2deg(skewer_angles['chosen_jaco_pitch_tilt_rad']):.1f}, "
                            f"Atool Pitch: {np.rad2deg(skewer_angles['calculated_atool_pitch_rad']):.1f}, "
                            f"Target Skewer Angle: {np.rad2deg(skewer_angles['skewer_polar_angle_rad']):.1f}"
                        )
                    planning_time = time.time() - start_time
                    status = (
                        TrialStatus.SUCCESS
                        if jaco_ee_above_food and jaco_ee_in_food
                        else TrialStatus.IK_FAILURE
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "PreAcquisition",
                            "status": status.value,
                            "planning_time_sec": planning_time,
                            "custom_metrics": skewer_angles if skewer_angles else {},
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        LOGGER.error(
                            f"  Stage 2 failed. Could not calculate required wrist poses."
                        )
                        trial_failed = True
                elif self.mode == "6dof_baseline":
                    LOGGER.info("Stage 2: Pre-acquisition")
                    start_time = time.time()
                    ik_above = self.motion_planner.compute_ik(
                        PLANNING_GROUP_JACO,
                        scene["above_food_pose"],
                        current_jaco_state,
                    )
                    ik_in = self.motion_planner.compute_ik(
                        PLANNING_GROUP_JACO,
                        scene["in_food_pose"],
                        current_jaco_state,
                    )
                    planning_time = time.time() - start_time
                    status = (
                        TrialStatus.SUCCESS
                        if ik_above and ik_in
                        else TrialStatus.IK_FAILURE
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "PreAcquisition",
                            "status": status.value,
                            "planning_time_sec": planning_time,
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        LOGGER.error(
                            f"  Stage 2 failed. Acquisition poses unreachable."
                        )
                        trial_failed = True
                elif self.mode == "8dof_baseline":
                    LOGGER.info("Stage 2: Pre-acquisition")
                    start_time = time.time()
                    found_valid_pair = False
                    # Try up to 10 times to find a valid elbow-up pair for both poses
                    for _ in range(10):
                        # Check AboveFood pose
                        ik_above_msg = self.motion_planner.compute_ik(
                            PLANNING_GROUP_FULL,
                            scene["above_food_pose"],
                            current_jaco_state + current_atool_state,
                        )
                        if (
                            not ik_above_msg
                            or not kinematic_solvers.is_elbow_up_configuration(
                                list(ik_above_msg.position)[:6], self.kinematics_model
                            )
                        ):
                            continue  # Try again if IK fails or is elbow-down

                        # If AboveFood is good, check InFood pose
                        ik_in_msg = self.motion_planner.compute_ik(
                            PLANNING_GROUP_FULL,
                            scene["in_food_pose"],
                            current_jaco_state + current_atool_state,
                        )
                        if ik_in_msg and kinematic_solvers.is_elbow_up_configuration(
                            list(ik_in_msg.position)[:6], self.kinematics_model
                        ):
                            # Found a pair where both are valid and elbow-up
                            LOGGER.info(
                                "    Found a valid 'elbow-up' IK solution pair."
                            )
                            found_valid_pair = True
                            break  # Success, exit the loop

                    planning_time = time.time() - start_time
                    status = (
                        TrialStatus.SUCCESS
                        if found_valid_pair
                        else TrialStatus.IK_FAILURE
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "PreAcquisition",
                            "status": status.value,
                            "planning_time_sec": planning_time,
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        LOGGER.error(
                            f"  Stage 2 failed. Acquisition poses unreachable."
                        )
                        trial_failed = True

            # --- Stage 3: AbovePlate -> AboveFood ---
            if not trial_failed:
                if self.mode == "articutool":
                    LOGGER.info("Stage 3: AbovePlate -> AboveFood")

                    # 1. Plan for the Jaco arm to the pre-calculated wrist pose
                    goal_constraints_jaco = [
                        create_pose_constraint(
                            jaco_ee_above_food, tolerance_position=0.01
                        )
                    ]
                    status_jaco, traj_jaco, planning_time_jaco = (
                        self.motion_planner.plan(
                            group_name=PLANNING_GROUP_JACO,
                            start_state=current_jaco_state,
                            goal_constraints=goal_constraints_jaco,
                            target_link=END_EFFECTOR_LINK_JACO,
                            planning_time=20.0,
                        )
                    )

                    # 2. If Jaco plan succeeds, plan for the Articutool
                    status_atool, traj_atool, planning_time_atool = (
                        TrialStatus.SKIPPED,
                        None,
                        0.0,
                    )
                    if status_jaco == TrialStatus.SUCCESS:
                        target_atool_config = [
                            skewer_angles["calculated_atool_pitch_rad"],
                            0.0,
                        ]
                        goal_constraints_atool = [
                            create_joint_constraint(target_atool_config)
                        ]
                        status_atool, traj_atool, planning_time_atool = (
                            self.motion_planner.plan(
                                group_name=PLANNING_GROUP_ATOOL,
                                start_state=current_atool_state,
                                goal_constraints=goal_constraints_atool,
                            )
                        )

                    # 3. Aggregate results and metrics
                    final_status = (
                        status_jaco
                        if status_jaco != TrialStatus.SUCCESS
                        else status_atool
                    )
                    planning_time = planning_time_jaco + planning_time_atool
                    cartesian_path_length_jaco = (
                        metrics.calculate_cartesian_path_length(
                            traj_jaco,
                            PLANNING_GROUP_JACO,
                            self.kinematics_model,
                            self.jaco_ee_link,
                        )
                    )
                    cartesian_path_length_atool = (
                        metrics.calculate_cartesian_path_length(
                            traj_atool,
                            PLANNING_GROUP_ATOOL,
                            self.kinematics_model,
                            self.jaco_ee_link,
                        )
                    )
                    cartesian_path_length = (
                        cartesian_path_length_jaco + cartesian_path_length_atool
                    )
                    joint_travel_jaco = metrics.calculate_total_joint_travel(traj_jaco)
                    joint_travel_atool = metrics.calculate_total_joint_travel(
                        traj_atool
                    )
                    joint_travel = joint_travel_jaco + joint_travel_atool

                    trial_data["stages"].append(
                        {
                            "stage_name": "AbovePlateToAboveFood",
                            "target_frame": END_EFFECTOR_LINK_FULL,
                            "status": final_status.value,
                            "execution_mode": ExecutionMode.SEQUENTIAL.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "custom_metrics": {},
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if final_status != TrialStatus.SUCCESS:
                        LOGGER.error(f"  Stage 3 failed. Skipping trial.")
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                        current_atool_state = list(traj_atool.points[-1].positions)
                elif self.mode == "6dof_baseline":
                    LOGGER.info("Stage 3: AbovePlate -> AboveFood")
                    goal_constraints = [
                        create_pose_constraint(
                            scene["above_food_pose"], tolerance_position=0.01
                        )
                    ]
                    status, traj_jaco, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_JACO,
                        start_state=current_jaco_state,
                        goal_constraints=goal_constraints,
                        planning_time=20.0,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    trial_data["stages"].append(
                        {
                            "stage_name": "AbovePlateToAboveFood",
                            "status": status.value,
                            "execution_mode": ExecutionMode.JACO_ONLY.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)

                elif self.mode == "8dof_baseline":
                    LOGGER.info("Stage 3: AbovePlate -> AboveFood")
                    goal_constraints = [
                        create_pose_constraint(
                            scene["above_food_pose"], tolerance_position=0.01
                        )
                    ]
                    status, traj_full, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_FULL,
                        start_state=current_jaco_state + current_atool_state,
                        goal_constraints=goal_constraints,
                        planning_time=20.0,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_full,
                        PLANNING_GROUP_FULL,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_full)
                    traj_jaco, traj_atool = trajectory_utils.split_full_trajectory(
                        traj_full
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "AbovePlateToAboveFood",
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                        current_atool_state = list(traj_atool.points[-1].positions)
            # --- Stage 4: AboveFood -> InFood ---
            if not trial_failed:
                if self.mode == "articutool":
                    LOGGER.info("Stage 4: AboveFood -> InFood (Cartesian)")
                    goal_constraints = [
                        create_pose_constraint(jaco_ee_in_food, tolerance_position=0.01)
                    ]
                    status, traj_jaco, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_JACO,
                        start_state=current_jaco_state,
                        goal_constraints=goal_constraints,
                        target_link=END_EFFECTOR_LINK_JACO,
                        cartesian=True,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)

                    # Create a hold trajectory for the Articutool with its target pitched angle
                    traj_atool = None
                    if status == TrialStatus.SUCCESS:
                        target_atool_config = [
                            skewer_angles["calculated_atool_pitch_rad"],
                            0.0,
                        ]
                        traj_atool = trajectory_utils.generate_hold_trajectory(
                            traj_jaco, JOINT_NAMES_ATOOL, target_atool_config
                        )

                    trial_data["stages"].append(
                        {
                            "stage_name": "AboveFoodToInFood",
                            "target_frame": END_EFFECTOR_LINK_FULL,
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "custom_metrics": {},
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        LOGGER.error(f"  Stage 4 failed. Skipping trial.")
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                        current_atool_state = list(traj_atool.points[-1].positions)
                elif self.mode == "6dof_baseline":
                    LOGGER.info("Stage 4: AboveFood -> InFood")
                    goal_constraints = [
                        create_pose_constraint(
                            scene["in_food_pose"], tolerance_position=0.01
                        )
                    ]
                    status, traj_jaco, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_JACO,
                        start_state=current_jaco_state,
                        goal_constraints=goal_constraints,
                        cartesian=True,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    trial_data["stages"].append(
                        {
                            "stage_name": "AboveFoodToInFood",
                            "status": status.value,
                            "execution_mode": ExecutionMode.JACO_ONLY.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                elif self.mode == "8dof_baseline":
                    LOGGER.info("Stage 4: AboveFood -> InFood")
                    goal_constraints = [
                        create_pose_constraint(
                            scene["in_food_pose"], tolerance_position=0.01
                        )
                    ]
                    status, traj_full, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_FULL,
                        start_state=current_jaco_state + current_atool_state,
                        goal_constraints=goal_constraints,
                        cartesian=True,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_full,
                        PLANNING_GROUP_FULL,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_full)
                    traj_jaco, traj_atool = trajectory_utils.split_full_trajectory(
                        traj_full
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "AboveFoodToInFood",
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                        current_atool_state = list(traj_atool.points[-1].positions)

            # --- Stage 5: InFood -> LevelTool ---
            if not trial_failed:
                if self.mode == "articutool":
                    LOGGER.info("Stage 5: Level Tool")
                    status, traj_atool, planning_time = self._plan_to_level_articutool(
                        jaco_ee_in_food, current_atool_state
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_atool,
                        PLANNING_GROUP_ATOOL,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_atool)
                    trial_data["stages"].append(
                        {
                            "stage_name": "LevelTool",
                            "target_frame": END_EFFECTOR_LINK_ATOOL,
                            "status": status.value,
                            "execution_mode": ExecutionMode.ATOOL_ONLY.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "custom_metrics": {},
                            "traj_jaco": None,
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        LOGGER.error(f"  Stage 5 failed. Skipping trial.")
                        trial_failed = True
                    else:
                        current_atool_state = list(traj_atool.points[-1].positions)
                elif self.mode == "6dof_baseline":
                    LOGGER.info("Stage 5: Level Tool")
                    status, traj_jaco, planning_time = (
                        self._plan_level_and_extract_baseline(
                            current_jaco_state,
                            scene["in_food_pose"],
                        )
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    trial_data["stages"].append(
                        {
                            "stage_name": "LevelTool",
                            "target_frame": self.jaco_ee_link,
                            "status": status.value,
                            "execution_mode": ExecutionMode.JACO_ONLY.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                elif self.mode == "8dof_baseline":
                    LOGGER.info("Stage 5: InFood -> LevelTool")
                    status, traj_full, planning_time = (
                        self._plan_level_and_extract_8dof_baseline(
                            current_jaco_state + current_atool_state,
                            scene["in_food_pose"],
                        )
                    )

                    # Split the 8-DOF trajectory for logging and state updates
                    traj_jaco, traj_atool = trajectory_utils.split_full_trajectory(
                        traj_full
                    )

                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_full,
                        PLANNING_GROUP_FULL,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_full)

                    trial_data["stages"].append(
                        {
                            "stage_name": "LevelTool",
                            "target_frame": END_EFFECTOR_LINK_FULL,
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                        current_atool_state = list(traj_atool.points[-1].positions)

            # --- Stage 6: LevelArticutool -> Resting ---
            if not trial_failed:
                if self.mode == "articutool":
                    LOGGER.info("Stage 6: Resting")
                    # First, validate the start state before attempting the real plan
                    is_start_state_valid = self._validate_start_state(
                        current_jaco_state, PLANNING_GROUP_JACO
                    )
                    if not is_start_state_valid:
                        # If the start state is invalid, log the specific failure and skip
                        status = TrialStatus.INVALID_START_STATE
                        traj_jaco, traj_atool, verification_results, planning_time = (
                            None,
                            None,
                            {},
                            0.0,
                        )
                    else:
                        (
                            status,
                            traj_jaco,
                            traj_atool,
                            verification_results,
                            planning_time,
                        ) = self._plan_to_resting(
                            scene["resting_pose"], current_jaco_state
                        )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel_jaco = metrics.calculate_total_joint_travel(traj_jaco)
                    joint_travel_atool = metrics.calculate_total_joint_travel(
                        traj_atool
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "Resting",
                            "target_frame": END_EFFECTOR_LINK_JACO,
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel_jaco
                            + joint_travel_atool,
                            "custom_metrics": verification_results,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        LOGGER.error(f"  Stage 7 failed. Skipping trial.")
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                        current_atool_state = list(traj_atool.points[-1].positions)
                elif self.mode == "6dof_baseline":
                    LOGGER.info("Stage 6: Resting")
                    goal_constraints = [
                        create_position_constraint(
                            scene["resting_pose"].position, tolerance_position=0.1
                        )
                    ]
                    path_constraints = [
                        create_orientation_path_constraint(
                            quat_xyzw=PATH_CONSTRAINT_QUAT_XYZW,
                            tolerance_rad=BASELINE_PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
                        )
                    ]
                    status, traj_jaco, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_JACO,
                        start_state=current_jaco_state,
                        goal_constraints=goal_constraints,
                        path_constraints=path_constraints,
                        target_link=END_EFFECTOR_LINK_JACO,
                        planning_time=20.0,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    trial_data["stages"].append(
                        {
                            "stage_name": "Resting",
                            "target_frame": self.jaco_ee_link,
                            "status": status.value,
                            "execution_mode": ExecutionMode.JACO_ONLY.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                elif self.mode == "8dof_baseline":
                    LOGGER.info("Stage 6: Resting")
                    goal_constraints = [
                        create_position_constraint(
                            scene["resting_pose"].position, tolerance_position=0.1
                        )
                    ]
                    path_constraints = [
                        create_orientation_path_constraint(
                            quat_xyzw=PATH_CONSTRAINT_QUAT_XYZW,
                            tolerance_rad=BASELINE_PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
                        )
                    ]
                    status, traj_full, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_FULL,
                        start_state=current_jaco_state + current_atool_state,
                        goal_constraints=goal_constraints,
                        path_constraints=path_constraints,
                        target_link=END_EFFECTOR_LINK_FULL,
                        planning_time=20.0,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_full,
                        PLANNING_GROUP_FULL,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_full)
                    traj_jaco, traj_atool = trajectory_utils.split_full_trajectory(
                        traj_full
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "Resting",
                            "target_frame": END_EFFECTOR_LINK_FULL,
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                    if status != TrialStatus.SUCCESS:
                        trial_failed = True
                    else:
                        current_jaco_state = list(traj_jaco.points[-1].positions)
                        current_atool_state = list(traj_atool.points[-1].positions)

            # --- Stage 7: Resting -> Staging ---
            if not trial_failed:
                if self.mode == "articutool":
                    LOGGER.info("Stage 7: Resting -> Staging")
                    (
                        status,
                        traj_jaco,
                        traj_atool,
                        verification_results,
                        planning_time,
                    ) = self._plan_to_staging(scene["staging_pose"], current_jaco_state)
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    trial_data["stages"].append(
                        {
                            "stage_name": "Staging",
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "custom_metrics": verification_results,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                elif self.mode == "6dof_baseline":
                    LOGGER.info("Stage 7: Resting -> Staging")
                    goal_constraints = [
                        create_pose_constraint(
                            scene["staging_pose"], tolerance_position=0.1
                        )
                    ]
                    path_constraints = [
                        create_orientation_path_constraint(
                            quat_xyzw=PATH_CONSTRAINT_QUAT_XYZW,
                            tolerance_rad=BASELINE_PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
                        )
                    ]
                    status, traj_jaco, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_JACO,
                        start_state=current_jaco_state,
                        goal_constraints=goal_constraints,
                        path_constraints=path_constraints,
                        target_link=self.jaco_ee_link,
                        planning_time=20.0,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    trial_data["stages"].append(
                        {
                            "stage_name": "Staging",
                            "status": status.value,
                            "execution_mode": ExecutionMode.JACO_ONLY.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": None,
                        }
                    )
                elif self.mode == "8dof_baseline":
                    LOGGER.info("Stage 7: Resting -> Staging")
                    goal_constraints = [
                        create_pose_constraint(
                            scene["staging_pose"], tolerance_position=0.1
                        )
                    ]
                    path_constraints = [
                        create_orientation_path_constraint(
                            quat_xyzw=PATH_CONSTRAINT_QUAT_XYZW,
                            tolerance_rad=BASELINE_PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
                        )
                    ]
                    status, traj_full, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_FULL,
                        start_state=current_jaco_state + current_atool_state,
                        goal_constraints=goal_constraints,
                        path_constraints=path_constraints,
                        target_link=END_EFFECTOR_LINK_FULL,
                        planning_time=20.0,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_full,
                        PLANNING_GROUP_FULL,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    traj_jaco, traj_atool = trajectory_utils.split_full_trajectory(
                        traj_full
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "Staging",
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )

                if status != TrialStatus.SUCCESS:
                    trial_failed = True
                else:
                    current_jaco_state = list(traj_jaco.points[-1].positions)
                    if self.mode in ["articutool", "8dof_baseline"] and traj_atool:
                        current_atool_state = list(traj_atool.points[-1].positions)

            # --- Stage 8: Staging -> Presentation ---
            if not trial_failed:
                if self.mode == "articutool":
                    LOGGER.info("Stage 8: Staging -> Presentation")
                    ik_sol_raw = self.motion_planner.compute_ik(
                        PLANNING_GROUP_FULL,
                        scene["presentation_pose"],
                        current_jaco_state + current_atool_state,
                    )
                    ik_sol_full = self._extract_ordered_joint_solution(ik_sol_raw)

                    if not ik_sol_full:
                        status = TrialStatus.IK_FAILURE
                        traj_jaco, traj_atool, planning_time = None, None, 0.0
                    else:
                        (
                            status,
                            traj_jaco,
                            traj_atool,
                            planning_time,
                        ) = self._plan_presentation_from_staging(
                            scene["presentation_pose"],
                            current_jaco_state,
                        )
                    cartesian_path_length_jaco = (
                        metrics.calculate_cartesian_path_length(
                            traj_jaco,
                            PLANNING_GROUP_JACO,
                            self.kinematics_model,
                            self.jaco_ee_link,
                        )
                    )
                    cartesian_path_length_atool = (
                        metrics.calculate_cartesian_path_length(
                            traj_atool,
                            PLANNING_GROUP_ATOOL,
                            self.kinematics_model,
                            self.jaco_ee_link,
                        )
                    )
                    joint_travel_jaco = metrics.calculate_total_joint_travel(traj_jaco)
                    joint_travel_atool = metrics.calculate_total_joint_travel(
                        traj_atool
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "Presentation",
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length_jaco
                            + cartesian_path_length_atool,
                            "total_joint_travel_rad": joint_travel_jaco
                            + joint_travel_atool,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )
                elif self.mode == "6dof_baseline":
                    LOGGER.info("Stage 8: Staging -> Presentation")
                    goal_constraints = [
                        create_pose_constraint(scene["presentation_pose"])
                    ]
                    status, traj_jaco, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_JACO,
                        start_state=current_jaco_state,
                        goal_constraints=goal_constraints,
                        cartesian=True,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_jaco,
                        PLANNING_GROUP_JACO,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    trial_data["stages"].append(
                        {
                            "stage_name": "Presentation",
                            "status": status.value,
                            "execution_mode": ExecutionMode.JACO_ONLY.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                        }
                    )
                elif self.mode == "8dof_baseline":
                    LOGGER.info("Stage 8: Staging -> Presentation")
                    goal_constraints = [
                        create_pose_constraint(scene["presentation_pose"])
                    ]
                    status, traj_full, planning_time = self.motion_planner.plan(
                        group_name=PLANNING_GROUP_FULL,
                        start_state=current_jaco_state + current_atool_state,
                        goal_constraints=goal_constraints,
                        cartesian=True,
                    )
                    cartesian_path_length = metrics.calculate_cartesian_path_length(
                        traj_full,
                        PLANNING_GROUP_FULL,
                        self.kinematics_model,
                        self.jaco_ee_link,
                    )
                    joint_travel = metrics.calculate_total_joint_travel(traj_jaco)
                    traj_jaco, traj_atool = trajectory_utils.split_full_trajectory(
                        traj_full
                    )
                    trial_data["stages"].append(
                        {
                            "stage_name": "Presentation",
                            "status": status.value,
                            "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                            "planning_time_sec": planning_time,
                            "trajectory_path_length_m": cartesian_path_length,
                            "total_joint_travel_rad": joint_travel,
                            "traj_jaco": results.serialize_trajectory(traj_jaco),
                            "traj_atool": results.serialize_trajectory(traj_atool),
                        }
                    )

                if status != TrialStatus.SUCCESS:
                    trial_failed = True

            # --- Finalize Trial ---
            if not trial_failed:
                trial_data["end_to_end_success"] = True

            results.save_trial_data(trial_data, self.output_filename)
            LOGGER.info(f"Trial {i} data saved to {self.output_filename}")

        LOGGER.info("Benchmark finished.")
