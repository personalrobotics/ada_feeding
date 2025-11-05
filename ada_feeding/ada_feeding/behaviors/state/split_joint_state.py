# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from sensor_msgs.msg import JointState
import py_trees

from typing import Any, Dict, List, Optional, Union
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class SplitJointState(BlackboardBehavior):
    """
    Splits a full sensor_msgs.msg.JointState message into separate
    outputs based on provided lists of joint names.

    This is a utility behavior and does not require any ROS components.
    It reads all inputs from the blackboard, processes them, and writes
    to the blackboard in a single 'update()' tick.
    """

    def __init__(
        self,
        name: str,
        ns: str = "/",
        inputs: Optional[Dict[str, Union[BlackboardKey, Any]]] = None,
        outputs: Optional[Dict[str, Optional[BlackboardKey]]] = None,
    ):
        super().__init__(name, ns=ns, inputs=inputs, outputs=outputs)
        # We need a logger, but this behavior doesn't create its own node.
        # We'll rely on the logger from the tree's main node,
        # which is typically passed during setup.
        self.node = None
        self.feedback_message = "SplitJointState: "

    def blackboard_inputs(
        self,
        full_joint_state: Union[BlackboardKey, JointState] = None,
        jaco_joint_names: Union[BlackboardKey, List[str]] = None,
        articutool_joint_names: Union[BlackboardKey, List[str]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        full_joint_state: The complete JointState message to be split.
        jaco_joint_names: A list of joint names for the Jaco arm output.
        articutool_joint_names: A list of joint names for the Articutool output.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        jaco_joint_state: Union[BlackboardKey, JointState] = None,
        jaco_joint_names: Union[BlackboardKey, List[str]] = None,
        jaco_joint_positions: Union[BlackboardKey, List[float]] = None,
        articutool_joint_positions: Union[BlackboardKey, List[float]] = None,
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        jaco_joint_state: The output JointState message for the Jaco arm.
        jaco_joint_names: The output list[str] of joint names for the Jaco arm.
        jaco_joint_positions: The output list[float] of joint positions for the Jaco arm.
        articutool_joint_positions: The output list[float] of joint positions
                                      for the Articutool.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        """
        Get the node from the tree manager for logging.
        """
        try:
            self.node = kwargs["node"]
        except KeyError:
            self.node = None
            self.feedback_message = "no 'node' in setup's kwargs"

    def update(self):
        """
        Reads the full joint state, splits it, and writes the outputs.
        """
        if self.node is None:
            # This is a fallback if setup fails, though it shouldn't.
            py_trees.logging.Logger(self.name).error(self.feedback_message)
            return py_trees.common.Status.FAILURE

        # --- 1. Read Inputs ---
        try:
            full_state = self.blackboard_get("full_joint_state")
            jaco_names_list = self.blackboard_get("jaco_joint_names")
            atool_names_list = self.blackboard_get("articutool_joint_names")
        except KeyError as e:
            self.node.get_logger().error(
                f"[{self.name}] Failed to get blackboard key: {e}"
            )
            return py_trees.common.Status.FAILURE

        if not full_state or not jaco_names_list or not atool_names_list:
            self.node.get_logger().error(f"[{self.name}] Missing blackboard inputs.")
            return py_trees.common.Status.FAILURE

        # --- 2. Create Lookup Dictionary ---
        try:
            position_lookup = dict(zip(full_state.name, full_state.position))
        except Exception as e:
            self.node.get_logger().error(
                f"[{self.name}] Could not create lookup from full_joint_state: {e}"
            )
            return py_trees.common.Status.FAILURE

        jaco_state_msg = JointState()
        jaco_state_msg.header = full_state.header  # Copy header
        atool_positions = []

        # --- 3. Process and Split ---
        try:
            # Process Jaco joints
            jaco_positions_list = []
            for name in jaco_names_list:
                jaco_positions_list.append(position_lookup[name])

            # Set the new list outputs
            self.blackboard_set("jaco_joint_names", jaco_names_list)
            self.blackboard_set("jaco_joint_positions", jaco_positions_list)

            # Build the JointState message
            jaco_state_msg.name = jaco_names_list
            jaco_state_msg.position = jaco_positions_list
            jaco_state_msg.velocity = [0.0] * len(jaco_names_list)
            jaco_state_msg.effort = [0.0] * len(jaco_names_list)

            # Process Articutool joints
            for name in atool_names_list:
                atool_positions.append(position_lookup[name])

        except KeyError as e:
            self.node.get_logger().error(
                f"[{self.name}] Joint name {e} not found in full_joint_state names: {full_state.name}"
            )
            return py_trees.common.Status.FAILURE
        except Exception as e:
            self.node.get_logger().error(
                f"[{self.name}] Error during joint processing: {e}"
            )
            return py_trees.common.Status.FAILURE

        # --- 4. Write Outputs ---
        self.blackboard_set("jaco_joint_state", jaco_state_msg)
        self.blackboard_set("articutool_joint_positions", atool_positions)

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        """
        No cleanup needed for this simple behavior.
        """
        pass
