#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (Add appropriate Copyright/License if desired)

"""
Defines the ExtractPoseComponents behavior, which separates Pose/PoseStamped
into its constituent parts.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Header # For outputting header if PoseStamped
from typing import Union, Optional, Dict, Any

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status

# Local imports (adjust paths as needed)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ExtractPoseComponents(BlackboardBehavior):
    """
    Extracts Position and Orientation components from an input Pose or PoseStamped
    object stored on the blackboard. Optionally outputs the Header if the input
    is PoseStamped.

    Returns SUCCESS if extraction is possible, FAILURE otherwise.
    """

    def blackboard_inputs(
        self,
        input_pose_object: Union[BlackboardKey, Pose, PoseStamped],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        input_pose_object: The blackboard key resolving to the Pose or PoseStamped message
                           from which to extract components.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        output_position: Optional[BlackboardKey] = None, # -> Optional[Point]
        output_orientation: Optional[BlackboardKey] = None, # -> Optional[Quaternion]
        output_header: Optional[BlackboardKey] = None, # -> Optional[Header]
        success: Optional[BlackboardKey] = None, # -> bool
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        output_position: Blackboard key to write the extracted geometry_msgs/Point.
        output_orientation: Blackboard key to write the extracted geometry_msgs/Quaternion.
        output_header: Optional key to write the std_msgs/Header if input was PoseStamped.
                       If input was Pose, this key will be set to None.
        success: Optional key to write boolean success flag (True if extraction succeeded).
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        """Get the node if needed (e.g., for logging)."""
        # pylint: disable=attribute-defined-outside-init
        try:
            # Node access is useful for logging via self.logger
            self.node: Node = kwargs['node']
        except KeyError:
            # Fallback logger if node isn't passed (though BlackboardBehavior likely handles this)
            self.logger.warning(f"[{self.name}] Node not explicitly provided via setup kwargs.")
            self.node = None

    @override
    def initialise(self) -> None:
        """Clear output keys on the blackboard before execution."""
        self.logger.debug(f"[{self.name}] Initializing and clearing outputs.")
        # Use blackboard_set helper which handles key existence checks based on blackboard_outputs registration
        self.blackboard_set("output_position", None)
        self.blackboard_set("output_orientation", None)
        self.blackboard_set("output_header", None)
        self.blackboard_set("success", False)


    @override
    def update(self) -> Status:
        """Read input, extract components based on type, write outputs."""
        extracted_pos: Optional[Point] = None
        extracted_ori: Optional[Quaternion] = None
        extracted_header: Optional[Header] = None
        op_success = False # Assume failure initially

        try:
            # Read the input object from the blackboard
            pose_obj = self.blackboard_get("input_pose_object")

            # Check the type and extract components
            if isinstance(pose_obj, PoseStamped):
                self.logger.debug(f"[{self.name}] Input is PoseStamped. Extracting pose and header.")
                extracted_pos = pose_obj.pose.position
                extracted_ori = pose_obj.pose.orientation
                extracted_header = pose_obj.header # Get the header
                op_success = True
            elif isinstance(pose_obj, Pose):
                self.logger.debug(f"[{self.name}] Input is Pose. Extracting components.")
                extracted_pos = pose_obj.position
                extracted_ori = pose_obj.orientation
                extracted_header = None # No header available for Pose type
                op_success = True
            else:
                # Log error if input is neither Pose nor PoseStamped
                self.logger.error(f"[{self.name}] Input 'input_pose_object' is not Pose or PoseStamped (type: {type(pose_obj)}). Cannot extract components.")
                op_success = False

        except KeyError as e:
            # Error if the input blackboard key doesn't exist
            self.logger.error(f"[{self.name}] Blackboard key error reading input: {e}")
            op_success = False
        except AttributeError as e:
             # Error if the object exists but doesn't have expected fields (e.g., None value)
             self.logger.error(f"[{self.name}] Attribute error accessing pose components (is input object None?): {e}")
             op_success = False
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"[{self.name}] Unexpected error extracting pose components: {e}")
            op_success = False

        # Write results (or None on failure) to blackboard output keys
        # blackboard_set handles checking if the output key was defined
        self.blackboard_set("output_position", extracted_pos)
        self.blackboard_set("output_orientation", extracted_ori)
        self.blackboard_set("output_header", extracted_header)
        self.blackboard_set("success", op_success)

        # Return SUCCESS or FAILURE based on whether extraction worked
        return Status.SUCCESS if op_success else Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        """Log termination."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
