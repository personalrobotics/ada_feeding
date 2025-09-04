# Standard imports
from enum import Enum, auto
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
import math

from .constants import ARTICUTOOL_LENGTH_M


# --- Data Structures ---
class TrialStatus(Enum):
    SUCCESS = "Success"
    IK_FAILURE = "IK Failure"
    PLANNER_FAILURE = "Planner Failure"
    VERIFICATION_FAILURE = "Path Verification Failure"
    INVALID_START_STATE = "Invalid Start State"
    SKIPPED = "Skipped"


class ExecutionMode(Enum):
    """Defines how trajectories for a stage should be interpreted."""

    SEQUENTIAL = "Sequential"
    SYNCHRONOUS = "Synchronous"
    JACO_ONLY = "Jaco Only"
    ATOOL_ONLY = "Articutool Only"


# --- Semantic Schema for Acquisition Actions ---
class AcquisitionStrategy(Enum):
    SKEWER = auto()
    SCOOP = auto()
    CUT = auto()


class MotionAxis(Enum):
    MAJOR_AXIS = auto()
    MINOR_AXIS = auto()
    VERTICAL = auto()


class ToolAlignment(Enum):
    PARALLEL = auto()
    PERPENDICULAR = auto()


@dataclass
class ActionRecipe:
    """Holds the semantic parameters for a single acquisition action"""

    strategy: AcquisitionStrategy
    motion_axis: MotionAxis
    tool_alignment: ToolAlignment

    def __str__(self):
        return (
            f"{self.strategy.name}-{self.motion_axis.name}-{self.tool_alignment.name}"
        )


@dataclass
class SphericalSamplingParams:
    """Parameters for sampling a pose within a spherical shell."""

    name: str
    inner_radius: float
    outer_radius: float
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]
    min_height: Optional[float] = None
    max_height: Optional[float] = None
    min_horizontal_radius: Optional[float] = None


@dataclass
class SceneGenerationParams:
    """
    Holds all parameters that define the random scene generation, tuned for a
    fair comparison within the 6-DOF baseline's workspace.
    """

    food_sampling: SphericalSamplingParams = SphericalSamplingParams(
        name="food",
        inner_radius=0.5,
        outer_radius=0.85,
        theta_range=(3 * math.pi / 2, 2 * math.pi),
        phi_range=(0.0, math.pi / 2),
        min_height=0.05,
        max_height=0.25,
        min_horizontal_radius=0.5,
    )
    mouth_sampling: SphericalSamplingParams = SphericalSamplingParams(
        name="mouth",
        inner_radius=0.4,
        outer_radius=0.6,
        theta_range=(0.0, math.pi / 4),
        phi_range=(0.0, math.pi / 2),
        min_height=0.3,
        max_height=0.7,
        min_horizontal_radius=0.3,
    )
    resting_sampling: SphericalSamplingParams = SphericalSamplingParams(
        name="resting",
        inner_radius=0.5,
        outer_radius=0.6,
        theta_range=(5 * math.pi / 4, 3 * math.pi / 2),
        phi_range=(0, math.pi / 2),
        min_height=0.2,
        max_height=0.4,
        min_horizontal_radius=0.5,
    )
    above_plate_radial_dist: float = 0.3
    above_plate_polar_angle_rad_max: float = math.pi / 3
    above_plate_yaw_variability_rad: float = math.pi / 8
    skewer_polar_angle_rad_max: float = math.pi / 4
    in_food_tool_roll_angle_deg: float = 180.0
    above_food_offset_dist: float = 0.05
    staging_offset_dist: float = ARTICUTOOL_LENGTH_M + 0.25
    presentation_offset_dist: float = 0.05
    # Defines how much to offset the mouth from an ideal staging pose.
    # 1.0 means the staging_pose will be at the initially sampled location.
    staging_scoot_back_factor: float = 1.0
