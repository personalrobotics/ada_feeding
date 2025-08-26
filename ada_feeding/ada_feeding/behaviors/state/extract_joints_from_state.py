# -*- coding: utf-8 -*-

# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the ExtractJointsFromState behavior, which extracts specific
joint names and their corresponding positions from a JointState message.
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

# Local imports (adjust paths as needed)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ExtractJointsFromState(BlackboardBehavior):
    """
    Extracts specified joint names and their corresponding positions from a
    source JointState message read from the blackboard. Writes the results
    as separate lists (names and positions) to the blackboard.
    """

    def blackboard_inputs(
        self,
        source_joint_state: Union[BlackboardKey, JointState],
        target_joint_names: Union[BlackboardKey, List[str]],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        source_joint_state: BlackboardKey resolving to the input JointState message
                            (e.g., the result of an IK computation).
        target_joint_names: BlackboardKey resolving to the List[str] of joint names
                            to extract from the source_joint_state.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        output_joint_names: Optional[BlackboardKey],  # -> Optional[List[str]]
        output_joint_positions: Optional[BlackboardKey],  # -> Optional[List[float]]
        success: Optional[BlackboardKey],  # -> bool (Optional success flag)
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        output_joint_names: BlackboardKey where the extracted List[str] of joint names
                            (in the order requested) will be written.
        output_joint_positions: BlackboardKey where the extracted List[float] of corresponding
                                joint positions will be written.
        success: Optional BlackboardKey to write a boolean indicating successful extraction.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def initialise(self) -> None:
        """Clear output keys on initialization."""
        self.logger.debug(f"[{self.name}] Initializing.")
        # Clear potential previous results
        self.blackboard_set("output_joint_names", None)
        self.blackboard_set("output_joint_positions", None)
        self.blackboard_set("success", False)

    @override
    def update(self) -> Status:
        """
        Reads inputs, performs extraction, writes outputs.
        Returns SUCCESS if all target joints are found, FAILURE otherwise.
        """
        try:
            source_state: JointState = self.blackboard_get("source_joint_state")
            target_names: List[str] = self.blackboard_get("target_joint_names")

            if not isinstance(source_state, JointState):
                self.logger.error(
                    f"[{self.name}] Input 'source_joint_state' is not a JointState message (is type: {type(source_state)})."
                )
                return Status.FAILURE

            if not isinstance(target_names, list) or not all(
                isinstance(n, str) for n in target_names
            ):
                self.logger.error(
                    f"[{self.name}] Input 'target_joint_names' must be a List[str]."
                )
                return Status.FAILURE

            if not target_names:
                self.logger.warning(
                    f"[{self.name}] Input 'target_joint_names' is empty. Nothing to extract."
                )
                self.blackboard_set("output_joint_names", [])
                self.blackboard_set("output_joint_positions", [])
                self.blackboard_set("success", True)
                return Status.SUCCESS

            # Create a lookup dictionary for efficient access
            source_positions_dict = {
                name: pos for name, pos in zip(source_state.name, source_state.position)
            }

            extracted_names = []
            extracted_positions = []
            all_found = True

            for name in target_names:
                if name in source_positions_dict:
                    extracted_names.append(name)
                    extracted_positions.append(source_positions_dict[name])
                else:
                    self.logger.error(
                        f"[{self.name}] Target joint '{name}' not found in source_joint_state names: {list(source_positions_dict.keys())}"
                    )
                    all_found = False
                    break

            if all_found:
                self.blackboard_set("output_joint_names", extracted_names)
                self.blackboard_set("output_joint_positions", extracted_positions)
                self.blackboard_set("success", True)
                self.logger.info(
                    f"[{self.name}] Successfully extracted {len(extracted_names)} joints."
                )
                return Status.SUCCESS
            else:
                self.blackboard_set("output_joint_names", None)
                self.blackboard_set("output_joint_positions", None)
                self.blackboard_set("success", False)
                return Status.FAILURE

        except KeyError as e:
            self.logger.error(f"[{self.name}] Blackboard key error: {e}")
            self.blackboard_set("success", False)
            return Status.FAILURE
        except Exception as e:
            self.logger.error(
                f"[{self.name}] Unexpected error during joint extraction: {e}",
                exc_info=True,
            )
            self.blackboard_set("success", False)
            return Status.FAILURE
