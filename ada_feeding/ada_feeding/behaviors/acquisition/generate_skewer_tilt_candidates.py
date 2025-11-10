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

    This version sorts the angles using a "median-out" approach,
    starting from the median angle and working outwards. This heuristics
    helps find a viable, non-extreme solution faster.
    """

    def blackboard_inputs(
        self,
        jaco_ee_tilt_angle_min: Optional[BlackboardKey],
        jaco_ee_tilt_angle_max: Optional[BlackboardKey],
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
        self.tilt_range_deg = (
            self.blackboard_get("jaco_ee_tilt_angle_min"),
            self.blackboard_get("jaco_ee_tilt_angle_max"),
            10.0,
        )

    @override
    def update(self) -> Status:
        """
        Generate the list of angles, sort them by distance from the median
        ("median-out"), write to the blackboard, and succeed.
        """
        self.feedback_message = "Generating skewer tilt candidates..."

        # 1. Generate angles in degrees
        candidate_tilts_deg = np.arange(*self.tilt_range_deg)

        if len(candidate_tilts_deg) == 0:
            self.logger.warning(
                f"[{self.name}] No tilt candidates generated with range {self.tilt_range_deg}. Failing."
            )
            self.feedback_message = "No candidates generated."
            return Status.FAILURE

        # 2. Find the median angle
        median_angle = np.median(candidate_tilts_deg)

        # 3. Sort by distance from the median ("median-out" heuristic)
        sorted_tilts_deg = sorted(
            candidate_tilts_deg, key=lambda x: abs(x - median_angle)
        )

        # 4. Convert to radians
        candidate_tilts_rad = [np.deg2rad(t) for t in sorted_tilts_deg]

        # 5. Write the list and the initial index to the blackboard
        self.blackboard_set("tilt_candidates_rad", candidate_tilts_rad)
        self.blackboard_set("tilt_index", 0)

        self.feedback_message = (
            f"Generated {len(candidate_tilts_rad)} candidates (median-out)."
        )
        self.logger.info(
            f"[{self.name}] Sorted tilt candidates (deg): {[round(t, 1) for t in sorted_tilts_deg]}"
        )
        return Status.SUCCESS
