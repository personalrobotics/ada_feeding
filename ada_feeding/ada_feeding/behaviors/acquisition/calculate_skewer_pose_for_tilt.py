# Standard Imports
import math
from typing import Union, Optional, List
from overrides import override

# Third-party Imports
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, TransformStamped
from scipy.spatial.transform import Rotation as R
import py_trees
from py_trees.common import Status
import pinocchio as pin

# Local BT Imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey

# Import benchmark constants for physical limits
from ada_feeding.articutool_benchmark.feeding_benchmark import (
    kinematics,
    constants as benchmark_constants,
)


def calculate_skewer_angle_from_pose(
    tool_tip_pose_world: Pose, food_frame_pose_world: Pose, logger
) -> float:
    """
    Calculates the tool's skewer polar angle as its deviation from vertical.
    """
    # --- DEBUG LOG 1: Print the raw input orientations ---
    q_tool = tool_tip_pose_world.orientation
    q_food = food_frame_pose_world.orientation
    logger.debug(
        f"    [Helper] Input Tool Quat (world): [x={q_tool.x:.3f}, y={q_tool.y:.3f}, z={q_tool.z:.3f}, w={q_tool.w:.3f}]"
    )
    logger.debug(
        f"    [Helper] Input Food Quat (world): [x={q_food.x:.3f}, y={q_food.y:.3f}, z={q_food.z:.3f}, w={q_food.w:.3f}]"
    )

    R_world_tool = R.from_quat([q_tool.x, q_tool.y, q_tool.z, q_tool.w])
    R_world_food = R.from_quat([q_food.x, q_food.y, q_food.z, q_food.w])

    R_food_tool = R_world_food.inv() * R_world_tool

    # --- DEBUG LOG 2: Print the calculated relative orientation ---
    q_food_tool = R_food_tool.as_quat()
    logger.debug(
        f"    [Helper] Relative Tool Quat (in food frame): [x={q_food_tool[0]:.3f}, y={q_food_tool[1]:.3f}, z={q_food_tool[2]:.3f}, w={q_food_tool[3]:.3f}]"
    )

    tool_forward_in_food_frame = R_food_tool.apply([0.0, 0.0, 1.0])

    # --- DEBUG LOG 3: Print the critical vector ---
    logger.debug(
        f"    [Helper] Tool +Z Vector (in food frame): [{tool_forward_in_food_frame[0]:.3f}, {tool_forward_in_food_frame[1]:.3f}, {tool_forward_in_food_frame[2]:.3f}]"
    )

    food_down_vector = np.array([0.0, 0.0, -1.0])
    dot_product = np.dot(tool_forward_in_food_frame, food_down_vector)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # --- DEBUG LOG 4: Print the dot product ---
    logger.debug(f"    [Helper] Dot Product with 'down' vector: {dot_product:.3f}")

    skewer_angle_rad = math.acos(dot_product)
    return skewer_angle_rad


class CalculateSkewerPoseForTilt(BlackboardBehavior):
    """
    Calculates candidate Jaco wrist poses to achieve a desired tool tip skewer motion.
    """

    def blackboard_inputs(
        self,
        tilt_candidates_rad: Union[BlackboardKey, list],
        tilt_index: Union[BlackboardKey, int],
        tool_tip_move_above_pose_world: Union[BlackboardKey, PoseStamped],
        tool_tip_move_into_pose_world: Union[BlackboardKey, PoseStamped],
        initial_food_frame: Union[BlackboardKey, TransformStamped],
        pinocchio_model: Union[BlackboardKey, pin.Model],
        pinocchio_data: Union[BlackboardKey, pin.Data],
        articutool_joint_names: Union[BlackboardKey, list],
    ) -> None:
        """Define blackboard inputs for this behavior."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        candidate_jaco_ee_above_pose: Optional[BlackboardKey],
        candidate_jaco_ee_into_pose: Optional[BlackboardKey],
        articutool_joint_positions: Union[BlackboardKey, List[float]],
        # This behavior also updates the index for the next iteration
        tilt_index: Optional[BlackboardKey],
    ) -> None:
        """Define blackboard outputs for this behavior."""
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def _get_relative_transform(
        self,
        model: pin.Model,
        data: pin.Data,
        parent_frame: str,
        child_frame: str,
        atool_joints: list,
        atool_joint_names: list,
    ) -> Optional[pin.SE3]:
        """
        Replicates the logic from the benchmark's PinocchioModel wrapper to
        compute a relative transform using raw pinocchio objects. This is
        used to find the transform from the Jaco EE to the tool tip (T_wrist_tip).
        """
        try:
            q = pin.neutral(model)
            # Populate only the articutool joints as they are the only ones that vary here
            for i, joint_name in enumerate(atool_joint_names):
                joint_id = model.getJointId(joint_name)
                joint_obj = model.joints[joint_id]
                angle = atool_joints[i]
                idx_q = joint_obj.idx_q
                if joint_obj.nq == 2:  # Revolute joint with cos/sin representation
                    q[idx_q : idx_q + 2] = [math.cos(angle), math.sin(angle)]
                else:
                    q[idx_q] = angle

            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            parent_id = model.getFrameId(parent_frame)
            T_world_parent = data.oMf[parent_id]
            child_id = model.getFrameId(child_frame)
            T_world_child = data.oMf[child_id]

            return T_world_parent.inverse() * T_world_child
        except Exception as e:
            self.logger.error(
                f"[{self.name}] Internal relative transform calculation failed: {e}"
            )
            return None

    @override
    def update(self) -> Status:
        """
        Calculate and validate geometry for the current tilt index.
        """
        try:
            # 1. Read inputs from the blackboard
            jaco_tilt_candidates = self.blackboard_get("tilt_candidates_rad")
            index = self.blackboard_get("tilt_index")

            # Check if we have exhausted all candidates
            if index >= len(jaco_tilt_candidates):
                self.feedback_message = "All tilt candidates have been exhausted."
                return Status.FAILURE

            jaco_pitch_tilt_rad = jaco_tilt_candidates[index]
            self.logger.info(
                f"Attempting Jaco EE tilt {np.rad2deg(jaco_pitch_tilt_rad):.1f} deg (index {index})."
            )
            self.feedback_message = f"Attempting tilt angle {np.rad2deg(jaco_pitch_tilt_rad):.1f} deg (index {index})."

            # Increment the index for the *next* run before any potential failure
            self.blackboard_set("tilt_index", index + 1)

            tool_tip_above_pose = self.blackboard_get(
                "tool_tip_move_above_pose_world"
            ).pose
            tool_tip_into_pose = self.blackboard_get(
                "tool_tip_move_into_pose_world"
            ).pose
            initial_food_frame = self.blackboard_get("initial_food_frame")
            kin_model = self.blackboard_get("pinocchio_model")
            kin_data = self.blackboard_get("pinocchio_data")
            atool_joint_names = self.blackboard_get("articutool_joint_names")

            # --- DEBUG LOG: Expose the static transform from wrist to tip ---
            static_wrist_to_tip = self._get_relative_transform(
                kin_model,
                kin_data,
                benchmark_constants.END_EFFECTOR_LINK_JACO,
                benchmark_constants.END_EFFECTOR_LINK_ATOOL,
                [0.0, 0.0],
                atool_joint_names,
            )
            if static_wrist_to_tip:
                static_quat = R.from_matrix(static_wrist_to_tip.rotation).as_quat()
                self.logger.debug(
                    f"    [DEBUG] Static Wrist-to-Tip Quat: [x={static_quat[0]:.3f}, y={static_quat[1]:.3f}, z={static_quat[2]:.3f}, w={static_quat[3]:.3f}]"
                )
            # --- END DEBUG LOG ---

            food_frame_pose = Pose()
            food_frame_pose.position = Point(
                x=initial_food_frame.transform.translation.x,
                y=initial_food_frame.transform.translation.y,
                z=initial_food_frame.transform.translation.z,
            )
            food_frame_pose.orientation = initial_food_frame.transform.rotation

            # Pass the behavior's logger to the helper function
            skewer_polar_angle_rad = calculate_skewer_angle_from_pose(
                tool_tip_into_pose, food_frame_pose, self.logger
            )

            required_atool_pitch = (
                (math.pi / 2.0) - skewer_polar_angle_rad - jaco_pitch_tilt_rad
            )

            self.logger.info(
                f"  Target Skewer Angle: {np.rad2deg(skewer_polar_angle_rad):.1f} deg | "
                f"Jaco Tilt: {np.rad2deg(jaco_pitch_tilt_rad):.1f} deg -> "
                f"Required Atool Pitch: {np.rad2deg(required_atool_pitch):.1f} deg"
            )

            if not (
                benchmark_constants.ARTICUTOOL_PITCH_LIMITS_RAD[0]
                - benchmark_constants.EPSILON
                <= required_atool_pitch
                <= benchmark_constants.ARTICUTOOL_PITCH_LIMITS_RAD[1]
                + benchmark_constants.EPSILON
            ):
                self.feedback_message = "Required Articutool pitch is out of limits."
                self.logger.warning(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE

            atool_config = [required_atool_pitch, 0.0]
            T_wrist_tip = self._get_relative_transform(
                kin_model,
                kin_data,
                benchmark_constants.END_EFFECTOR_LINK_JACO,
                benchmark_constants.END_EFFECTOR_LINK_ATOOL,
                atool_config,
                atool_joint_names,
            )
            if T_wrist_tip is None:
                self.feedback_message = "Failed to calculate T_wrist_tip transform."
                return Status.FAILURE

            q_tip = tool_tip_into_pose.orientation
            R_world_tip = R.from_quat([q_tip.x, q_tip.y, q_tip.z, q_tip.w])

            R_wrist_tip = R.from_matrix(T_wrist_tip.rotation)
            R_world_wrist_untilted = R_world_tip * R_wrist_tip.inv()

            R_tilt = R.from_euler("x", jaco_pitch_tilt_rad)
            R_world_wrist_target = R_world_wrist_untilted * R_tilt

            p_tip_infood = np.array(
                [
                    tool_tip_into_pose.position.x,
                    tool_tip_into_pose.position.y,
                    tool_tip_into_pose.position.z,
                ]
            )
            p_wrist_infood = p_tip_infood - R_world_wrist_target.apply(
                T_wrist_tip.translation
            )

            p_tip_above = np.array(
                [
                    tool_tip_above_pose.position.x,
                    tool_tip_above_pose.position.y,
                    tool_tip_above_pose.position.z,
                ]
            )
            p_wrist_above = p_tip_above - R_world_wrist_target.apply(
                T_wrist_tip.translation
            )

            quat_xyzw = R_world_wrist_target.as_quat()
            final_orientation = Quaternion(
                x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2], w=quat_xyzw[3]
            )
            in_food_wrist_pose = Pose(
                position=Point(
                    x=p_wrist_infood[0], y=p_wrist_infood[1], z=p_wrist_infood[2]
                ),
                orientation=final_orientation,
            )
            above_food_wrist_pose = Pose(
                position=Point(
                    x=p_wrist_above[0], y=p_wrist_above[1], z=p_wrist_above[2]
                ),
                orientation=final_orientation,
            )

            # Write successful candidate poses to blackboard
            self.feedback_message = "Geometrically valid candidate found."
            self.blackboard_set("candidate_jaco_ee_above_pose", above_food_wrist_pose)
            self.blackboard_set("candidate_jaco_ee_into_pose", in_food_wrist_pose)
            self.blackboard_set("articutool_joint_positions", atool_config)

            return Status.SUCCESS

        except Exception as e:
            self.logger.error(f"[{self.name}] Exception: {e}")
            return Status.FAILURE
