# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the ExtractPoseFromPosesByLink behavior, which finds and extracts
a specific PoseStamped from a list based on its associated link name.
"""

# Standard imports
from typing import Union, Optional, Dict, Any, List

# Third-party imports
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status
import rclpy.node

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ExtractPoseFromPosesByLink(BlackboardBehavior):
    """
    Extracts a specific PoseStamped message from blackboard data which could be
    either a single PoseStamped or a list of PoseStamped messages.

    If the input is a list, it requires a corresponding list of link names (which
    should be the same list passed to the FK behavior that produced the pose list)
    and a target link name to identify which pose to extract.

    If the input is a single PoseStamped, it checks if the target link name matches
    expectations (either the single link name provided to FK or the default EE).
    """

    def blackboard_inputs(
        self,
        fk_poses: Union[BlackboardKey, PoseStamped, List[PoseStamped]],
        target_link_name: Union[BlackboardKey, str],
        requested_link_names: Optional[Union[BlackboardKey, List[str]]] = None,
        default_ee_link_name: Optional[Union[BlackboardKey, str]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        fk_poses: BlackboardKey resolving to the output of MoveIt2ComputeFK
                   (either PoseStamped or List[PoseStamped]).
        requested_link_names: BlackboardKey resolving to the List[str] that was passed
                              to MoveIt2ComputeFK's fk_link_names input. This is
                              REQUIRED if fk_poses is a List. Ignored if fk_poses
                              is a single PoseStamped.
        target_link_name: BlackboardKey resolving to the string name of the link whose
                          pose should be extracted.
        default_ee_link_name: Optional: BlackboardKey resolving to the default EE link name.
                              Used for validation when fk_poses is a single PoseStamped
                              and requested_link_names was empty/None.
        """
        super().blackboard_inputs(
            **{
                key: value
                for key, value in locals().items()
                if key not in ["self", "kwargs"]
            }
        )

    def blackboard_outputs(
        self,
        extracted_pose: Optional[BlackboardKey],  # -> Optional[PoseStamped]
        success: Optional[BlackboardKey],  # -> bool
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        extracted_pose: BlackboardKey where the single extracted PoseStamped will be written.
        success: Boolean flag indicating successful extraction.
        """
        super().blackboard_outputs(
            **{
                key: value
                for key, value in locals().items()
                if key not in ["self", "kwargs"]
            }
        )

    @override
    def initialise(self) -> None:
        """Clear output keys."""
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("extracted_pose", None)
        self.blackboard_set("success", False)

    @override
    def update(self) -> Status:
        """Extract the pose corresponding to the target link name."""
        try:
            fk_data = self.blackboard_get("fk_poses")
            target_link = self.blackboard_get("target_link_name")
            req_link_names = self.blackboard_get("requested_link_names")
            default_ee = self.blackboard_get("default_ee_link_name")

            extracted_pose_result: Optional[PoseStamped] = None
            found = False

            # Case 1: FK result is a single PoseStamped
            if isinstance(fk_data, PoseStamped):
                self.logger.debug(f"[{self.name}] FK result is single PoseStamped.")
                # Check if the target link matches expectations
                if req_link_names is None or (
                    isinstance(req_link_names, list) and len(req_link_names) == 0
                ):
                    # FK was likely called for the default EE
                    if default_ee is not None and target_link == default_ee:
                        extracted_pose_result = fk_data
                        found = True
                    elif default_ee is None:
                        self.logger.warning(
                            f"[{self.name}] Input fk_poses is single PoseStamped, but default_ee_link_name not provided for validation."
                        )
                        # Assume it's correct if only one link could have been requested
                        extracted_pose_result = fk_data
                        found = True
                    else:
                        self.logger.error(
                            f"[{self.name}] Target link '{target_link}' does not match default EE '{default_ee}' for single FK result."
                        )
                elif isinstance(req_link_names, list) and len(req_link_names) == 1:
                    # FK was called for one specific link
                    if target_link == req_link_names[0]:
                        extracted_pose_result = fk_data
                        found = True
                    else:
                        self.logger.error(
                            f"[{self.name}] Target link '{target_link}' does not match requested link '{req_link_names[0]}' for single FK result."
                        )
                else:
                    self.logger.error(
                        f"[{self.name}] Input fk_poses is single PoseStamped, but requested_link_names is ambiguous: {req_link_names}"
                    )

            # Case 2: FK result is a List of PoseStamped
            elif isinstance(fk_data, list):
                self.logger.debug(f"[{self.name}] FK result is a list.")
                if not isinstance(req_link_names, list) or len(fk_data) != len(
                    req_link_names
                ):
                    self.logger.error(
                        f"[{self.name}] 'fk_poses' is a list, but 'requested_link_names' is missing, not a list, or lengths mismatch (Poses: {len(fk_data)}, Names: {len(req_link_names or [])})."
                    )
                    return self._handle_failure()

                # Find the index matching the target link name
                try:
                    target_index = req_link_names.index(target_link)
                    if isinstance(fk_data[target_index], PoseStamped):
                        extracted_pose_result = fk_data[target_index]
                        found = True
                    else:
                        self.logger.error(
                            f"[{self.name}] Data at index {target_index} for link '{target_link}' is not a PoseStamped."
                        )
                except ValueError:
                    self.logger.error(
                        f"[{self.name}] Target link '{target_link}' not found in requested_link_names: {req_link_names}"
                    )
                except IndexError:
                    self.logger.error(
                        f"[{self.name}] Index mismatch error after finding target link '{target_link}'."
                    )

            # Case 3: FK result is None or unexpected type
            else:
                self.logger.error(
                    f"[{self.name}] Input 'fk_poses' is None or unexpected type ({type(fk_data)}). FK likely failed."
                )
                return self._handle_failure()

            # Write results and return status
            if found and extracted_pose_result is not None:
                self.blackboard_set("extracted_pose", extracted_pose_result)
                self.blackboard_set("success", True)
                self.logger.info(
                    f"[{self.name}] Successfully extracted pose for link '{target_link}'."
                )
                return Status.SUCCESS
            else:
                # Error logged within logic above
                return self._handle_failure()

        except KeyError as e:
            self.logger.error(f"[{self.name}] Blackboard key error: {e}")
            return self._handle_failure()
        except Exception as e:
            self.logger.error(f"[{self.name}] Unexpected error extracting pose: {e}")
            return self._handle_failure()

    def _handle_failure(self) -> Status:
        """Helper to set outputs on failure."""
        self.blackboard_set("extracted_pose", None)
        self.blackboard_set("success", False)
        return Status.FAILURE
