# -*- coding: utf-8 -*-
# (Add appropriate Copyright/License if desired)

"""
Defines the CombineJointStates behavior, which merges two JointState
messages into a single one based on a provided full list of joint names.
"""

# Standard imports
from typing import Union, Optional, Dict, Any, List

# Third-party imports
from sensor_msgs.msg import JointState
from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status
import rclpy.node
import rclpy.time

# Local imports (adjust paths as needed)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class CombineJointStates(BlackboardBehavior):
    """
    Merges two input JointState messages (e.g., from separate arm and tool plans)
    into a single output JointState message, ordered according to a provided
    list of all joint names for the combined system.
    """

    def blackboard_inputs(
        self,
        joint_state_1: Union[BlackboardKey, JointState],
        joint_state_2: Union[BlackboardKey, JointState],
        full_joint_names: Union[BlackboardKey, List[str]],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        joint_state_1: BlackboardKey resolving to the first JointState message.
        joint_state_2: BlackboardKey resolving to the second JointState message.
        full_joint_names: BlackboardKey resolving to the List[str] containing all
                          joint names for the combined state, in the desired order.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        combined_joint_state: Optional[BlackboardKey], # -> Optional[JointState]
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        combined_joint_state: BlackboardKey where the resulting merged JointState
                              message will be written.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        """Gets the ROS2 node from the arguments passed by the tree runner."""
        # pylint: disable=attribute-defined-outside-init
        try:
            self.node: rclpy.node.Node = kwargs['node']
        except KeyError as e:
            self.logger.error(f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}")
            self.node = None

    # initialise is not strictly needed, but good practice
    @override
    def initialise(self) -> None:
        """Optionally log initialization."""
        self.logger.debug(f"[{self.name}] Initializing.")
        # Clear previous output
        self.blackboard_set("combined_joint_state", None)


    @override
    def update(self) -> Status:
        """
        Reads input JointStates, merges them based on full_joint_names,
        and writes the combined JointState.
        """
        if self.node is None:
            self.logger.error(f"[{self.name}] Node object not available. Setup likely failed.")
            return Status.FAILURE

        try:
            # Read inputs
            state1: JointState = self.blackboard_get("joint_state_1")
            state2: JointState = self.blackboard_get("joint_state_2")
            full_names: List[str] = self.blackboard_get("full_joint_names")

            # --- Input Validation ---
            if not isinstance(state1, JointState):
                 self.logger.error(f"[{self.name}] Input 'joint_state_1' is not a JointState message (is type: {type(state1)}).")
                 return Status.FAILURE
            if not isinstance(state2, JointState):
                 self.logger.error(f"[{self.name}] Input 'joint_state_2' is not a JointState message (is type: {type(state2)}).")
                 return Status.FAILURE
            if not isinstance(full_names, list) or not all(isinstance(n, str) for n in full_names):
                 self.logger.error(f"[{self.name}] Input 'full_joint_names' must be a List[str].")
                 return Status.FAILURE
            if not full_names:
                 self.logger.error(f"[{self.name}] Input 'full_joint_names' cannot be empty.")
                 return Status.FAILURE
            # --- End Validation ---


            # --- Merge Logic ---
            # Create a dictionary to hold combined positions, giving priority to state2 if names overlap
            combined_positions_dict = {}
            for name, pos in zip(state1.name, state1.position):
                combined_positions_dict[name] = pos
            for name, pos in zip(state2.name, state2.position):
                combined_positions_dict[name] = pos # Overwrites if name was in state1

            # Create the output JointState
            output_state = JointState()
            output_state.header.stamp = self.node.get_clock().now().to_msg() # Use current time
            output_state.name = full_names # Assign the full ordered list of names
            output_state.position = [] # Initialize empty list

            # Populate positions in the order specified by full_names
            all_found = True
            for name in full_names:
                if name in combined_positions_dict:
                    output_state.position.append(combined_positions_dict[name])
                else:
                    self.logger.error(f"[{self.name}] Joint '{name}' from 'full_joint_names' not found in either input JointState.")
                    all_found = False
                    break # Stop processing

            if not all_found:
                return Status.FAILURE

            # Basic check for consistency
            if len(output_state.name) != len(output_state.position):
                 self.logger.error(f"[{self.name}] Mismatch between output names ({len(output_state.name)}) and positions ({len(output_state.position)}). Logic error?")
                 return Status.FAILURE
            # --- End Merge Logic ---


            # Write the successful result to the blackboard
            self.blackboard_set("combined_joint_state", output_state)
            self.logger.info(f"[{self.name}] Successfully combined joint states for {len(output_state.name)} joints.")
            return Status.SUCCESS

        except KeyError as e:
            self.logger.error(f"[{self.name}] Blackboard key error: {e}")
            return Status.FAILURE
        except Exception as e:
            self.logger.error(f"[{self.name}] Unexpected error combining joint states: {e}", exc_info=True)
            return Status.FAILURE
