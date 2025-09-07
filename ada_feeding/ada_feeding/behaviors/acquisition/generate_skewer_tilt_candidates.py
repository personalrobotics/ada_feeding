# Standard Imports
from overrides import override
from typing import Union, Optional

# Third-party Imports
import numpy as np
import py_trees
from py_trees.common import Status

# Local BT Imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class GenerateSkewerTiltCandidates(BlackboardBehavior):
    """
    Generates a list of candidate Jaco pitch tilt angles for the skewer motion.

    This behavior runs once, creating the list of angles to be tested and
    initializing an index on the blackboard to track the current attempt.
    """

    def blackboard_inputs(
        self,
    ) -> None:
        """Define blackboard inputs for this behavior."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        tilt_candidates_rad: Optional[BlackboardKey],
        tilt_index: Optional[BlackboardKey],
    ) -> None:
        """Define blackboard outputs for this behavior."""
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def initialise(self):
        """
        Initialize with the desired range of tilt angles in degrees.
        Format is (start, stop, step).
        """
        self.tilt_range_deg = (0.0, -50.0, -5.0)

    @override
    def update(self) -> Status:
        """
        Generate the list of angles, write to the blackboard, and succeed.
        """
        self.feedback_message = "Generating skewer tilt candidates..."

        # Generate angles in degrees and convert to radians
        candidate_tilts_deg = np.arange(*self.tilt_range_deg)
        candidate_tilts_rad = [np.deg2rad(t) for t in candidate_tilts_deg]

        # Write the list and the initial index to the blackboard
        self.blackboard_set("tilt_candidates_rad", candidate_tilts_rad)
        self.blackboard_set("tilt_index", 0)

        self.feedback_message = f"Generated {len(candidate_tilts_rad)} candidates."
        return Status.SUCCESS
