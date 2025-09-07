# Standard Imports
import math
from typing import Union, Optional, List
from overrides import override

# Third-party Imports
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import py_trees
from py_trees.common import Status
import pinocchio as pin

# Local BT Imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey

# Import your benchmark modules directly
from ada_feeding.articutool_benchmark.feeding_benchmark import (
    kinematics,
    constants as benchmark_constants,
)


def calculate_skewer_angle_from_pose(pose: PoseStamped) -> float:
    """Calculates the polar angle of a tool's skewer motion from its world-frame pose."""
    q = pose.pose.orientation
    world_from_tool_rotation = R.from_quat([q.x, q.y, q.z, q.w])
    tool_approach_vector_local = np.array([0.0, 0.0, 1.0])
    tool_approach_vector_world = world_from_tool_rotation.apply(
        tool_approach_vector_local
    )
    world_vertical_vector = np.array([0.0, 0.0, 1.0])
    dot_product = np.dot(tool_approach_vector_world, world_vertical_vector)
    angle_rad = math.acos(np.clip(dot_product, -1.0, 1.0))
    return angle_rad


class CalculateSkewerPoseForTilt(BlackboardBehavior):
    """
    Calculates the candidate Jaco wrist poses for a single tilt angle from a
    pre-generated list on the blackboard.

    This behavior is intended to be used within a Retry decorator. It reads an
    index, processes the corresponding tilt angle, and returns FAILURE if the
    geometry for that angle is invalid (e.g., Articutool pitch out of limits).
    On success, it writes the candidate poses to the blackboard for a subsequent
    IK check. It always increments the index for the next iteration.
    """

    def blackboard_inputs(
        self,
        tilt_candidates_rad: Union[BlackboardKey, list],
        tilt_index: Union[BlackboardKey, int],
        tool_tip_move_above_pose_world: Union[BlackboardKey, PoseStamped],
        tool_tip_move_into_pose_world: Union[BlackboardKey, PoseStamped],
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
        compute a relative transform using raw pinocchio objects.
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
            tilt_candidates = self.blackboard_get("tilt_candidates_rad")
            index = self.blackboard_get("tilt_index")

            # Check if we have exhausted all candidates
            if index >= len(tilt_candidates):
                self.feedback_message = "All tilt candidates have been exhausted."
                return Status.FAILURE

            jaco_pitch_tilt = tilt_candidates[index]
            self.logger.info(
                f"Attempting tilt angle {np.rad2deg(jaco_pitch_tilt):.1f} deg (index {index})."
            )
            self.feedback_message = f"Attempting tilt angle {np.rad2deg(jaco_pitch_tilt):.1f} deg (index {index})."

            # Increment the index for the *next* run before any potential failure
            self.blackboard_set("tilt_index", index + 1)

            tool_tip_above_pose = self.blackboard_get("tool_tip_move_above_pose_world")
            tool_tip_into_pose = self.blackboard_get("tool_tip_move_into_pose_world")
            kin_model = self.blackboard_get("pinocchio_model")
            kin_data = self.blackboard_get("pinocchio_data")
            atool_joint_names = self.blackboard_get("articutool_joint_names")
            skewer_polar_angle = calculate_skewer_angle_from_pose(tool_tip_into_pose)
            self.feedback_message += (
                f" Derived skewer angle: {np.rad2deg(skewer_polar_angle):.1f} deg."
            )
            # 2. Perform the geometric calculations from the benchmark solver
            required_atool_pitch = (
                (math.pi / 2.0) - skewer_polar_angle - jaco_pitch_tilt
            )
            if not (
                benchmark_constants.ARTICUTOOL_PITCH_LIMITS_RAD[0]
                - benchmark_constants.EPSILON
                <= required_atool_pitch
                <= benchmark_constants.ARTICUTOOL_PITCH_LIMITS_RAD[1]
                + benchmark_constants.EPSILON
            ):
                self.feedback_message = (
                    "Required Articutool pitch is out of limits for this tilt."
                )
                self.logger.warning(
                    f"[{self.name}] Tilt {np.rad2deg(jaco_pitch_tilt):.1f} deg failed: required pitch {np.rad2deg(required_atool_pitch):.1f} deg is outside limits."
                )
                return Status.FAILURE  # This will trigger the parent Retry decorator

            # Call the new internal helper method instead of the old wrapper
            T_wrist_tip = self._get_relative_transform(
                model=kin_model,
                data=kin_data,
                parent_frame=benchmark_constants.END_EFFECTOR_LINK_JACO,
                child_frame=benchmark_constants.END_EFFECTOR_LINK_ATOOL,
                atool_joints=[required_atool_pitch, 0.0],
                atool_joint_names=atool_joint_names,
            )
            if T_wrist_tip is None:
                self.feedback_message = "Failed to calculate T_wrist_tip transform."
                return Status.FAILURE  # This will trigger the parent Retry decorator

            # --- This is the geometric logic replicated from your benchmark solver ---
            q_tip = tool_tip_into_pose.pose.orientation
            R_world_tip = R.from_quat([q_tip.x, q_tip.y, q_tip.z, q_tip.w])
            R_wrist_tip = R.from_matrix(T_wrist_tip.rotation)
            R_world_wrist_untilted = R_world_tip * R_wrist_tip.inv()
            R_tilt = R.from_euler("x", jaco_pitch_tilt)
            R_world_wrist_target = R_world_wrist_untilted * R_tilt

            p_tip_infood = np.array(
                [
                    tool_tip_into_pose.pose.position.x,
                    tool_tip_into_pose.pose.position.y,
                    tool_tip_into_pose.pose.position.z,
                ]
            )
            p_wrist_infood = p_tip_infood - R_world_wrist_target.apply(
                T_wrist_tip.translation
            )
            q_target = R_world_wrist_target.as_quat()
            final_orientation = Quaternion(
                x=q_target[0], y=q_target[1], z=q_target[2], w=q_target[3]
            )
            in_food_wrist_pose = Pose(
                position=Point(
                    x=p_wrist_infood[0], y=p_wrist_infood[1], z=p_wrist_infood[2]
                ),
                orientation=final_orientation,
            )

            p_tip_above = np.array(
                [
                    tool_tip_above_pose.pose.position.x,
                    tool_tip_above_pose.pose.position.y,
                    tool_tip_above_pose.pose.position.z,
                ]
            )
            p_wrist_above = p_tip_above - R_world_wrist_target.apply(
                T_wrist_tip.translation
            )
            above_food_wrist_pose = Pose(
                position=Point(
                    x=p_wrist_above[0], y=p_wrist_above[1], z=p_wrist_above[2]
                ),
                orientation=final_orientation,
            )
            # --- End of replicated logic ---

            # 3. Write successful candidate poses to blackboard
            self.feedback_message = "Geometrically valid candidate found."
            self.blackboard_set("candidate_jaco_ee_above_pose", above_food_wrist_pose)
            self.blackboard_set("candidate_jaco_ee_into_pose", in_food_wrist_pose)
            self.blackboard_set(
                "articutool_joint_positions", [required_atool_pitch, 0.0]
            )

            return Status.SUCCESS

        except Exception as e:
            self.logger.error(f"[{self.name}] Exception: {e}")
            return Status.FAILURE
