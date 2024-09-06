#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the servo_until idiom, which takes in a condition behavior
that should never return RUNNING, and an instance of the ServoMove behavior and
returns a behavior that runs the ServoMove behavior until either it succeeds
(e.g., the ServoMove behavior times out) or the condition behavior returns
SUCCESS.

Note that in expected usage, the condition behavior will write a twist to the
blackboard, which will be read by the ServoMove behavior.
"""

# Standard imports
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Third-party imports
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Vector3,
)
import numpy as np
import py_trees
from py_trees.behaviours import UnsetBlackboardVariable
from py_trees.blackboard import Blackboard
from rclpy.duration import Duration
from rclpy.time import Time
import ros2_numpy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header

# Local imports
from ada_feeding.behaviors.moveit2 import ServoMove
from ada_feeding.behaviors.ros import (
    ApplyTransform,
    PoseStampedToTwistStamped,
    SetStaticTransform,
    TrackHz,
    UpdateTimestamp,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms.ft_thresh_utils import ft_thresh_satisfied


# Hardcoded parts of the behavior names that compute the distance for Servoing.
# This is used by MoveToVisitor to populate feedback. Try to make these as
# unique as possible to avoid clashes.
SERVO_UNTIL_POSE_DISTANCE_BEHAVIOR_NAME = "ServoUntilPose PoseStampedToTwistStamped"


def servo_until(
    name: str,
    ns: str,
    sense: py_trees.behaviour.Behaviour,
    check: py_trees.behaviour.Behaviour,
    compute_twist: py_trees.behaviour.Behaviour,
    servo_inputs: Dict[str, Union[BlackboardKey, Any]],
    f_mag_threshold: Optional[Union[BlackboardKey, float]] = None,
    t_mag_threshold: Optional[Union[BlackboardKey, float]] = None,
) -> py_trees.behaviour.Behaviour:
    """
    An idiom to implement servoing until a condition is met. Sepcifically, this
    behavior wll iteratively sense the world, check if a condition is met, if so
    return SUCCESS, else compute a twist and send it to the servo controller.

    Parameters
    ----------
    name: The name of the behavior.
    ns: The namespace of the action server.
    sense: A behavior that senses the world and writes to the blackboard.
    check: A behavior that checks whether the condition is met, based on the
        blackboard output of the `sense` behavior. Note that this behavior
        should never return RUNNING, and should never return FAILURE unless the
        condition is not yet met.
    compute_twist: A behavior that computes the twist to send to the servo
        controller, based on the blackboard output of the `sense` behavior.
    servo_inputs: Additional keyword arguments to pass as inputs to the servo
        controller.
    f_mag_threshold: If not None, then subscribe to the FT sensor and fail if the
        threshold is exceeded. If None, then do not subscribe to the FT sensor.
    t_mag_threshold: If not None, then subscribe to the FT sensor and fail if the
        threshold is exceeded. If None, then do not subscribe to the FT sensor.

    Returns
    -------
    A behavior that runs the servo_move behavior until the condition is met.
    """

    # pylint: disable=too-many-arguments
    # This is intended to be a flexible idiom.

    # Get the children of the servo sequence
    servo_children = []
    # if f_mag_threshold is not None or t_mag_threshold is not None:
    #     # Check the F/T threshold.
    #     kwargs = {
    #         "name": f"{name} CheckFT",
    #         "ns": ns,
    #     }
    #     if f_mag_threshold is not None:
    #         kwargs["f_mag"] = f_mag_threshold
    #     if t_mag_threshold is not None:
    #         kwargs["t_mag"] = t_mag_threshold
    #     # NOTE: Using `ft_thresh_satisfied` might be inefficient if the tree this
    #     # idiom is used in has other behaviors that subscribe to the F/T threshold.
    #     servo_children.append(ft_thresh_satisfied(**kwargs))
    servo_children.append(compute_twist)
    servo_children.append(ServoMove(name=f"{name} Servo", ns=ns, inputs=servo_inputs))

    return py_trees.composites.Sequence(
        name=name,
        memory=False,
        children=[
            sense,
            py_trees.composites.Selector(
                name=f"{name} Selector",
                memory=False,
                children=[
                    check,
                    py_trees.composites.Sequence(
                        name=f"{name} Sequence",
                        memory=False,
                        children=servo_children,
                    ),
                ],
            ),
        ],
    )


def pose_within_tolerances(
    current_pose: PoseStamped,
    target_pose: PoseStamped,
    tolerance_position: float = 0.001,
    tolerance_orientation: Union[float, Tuple[float, float, float]] = 0.001,
) -> bool:
    """
    Returns whether the current_pose is within the specified tolerances of the
    target_pose.

    Parameters
    ----------
    current_pose: The current pose.
    target_pose: The target pose.
    tolerance_position: The tolerance on the position.
    tolerance_orientation: The tolerance on the orientation. If a float, the
        tolerance is the same for all axes. This computes orientation tolerance
        in the same way MoveIt2 does for the "rotation vector" parameterization:
        https://github.com/ros-planning/moveit2/blob/f9989626491ca63e9f7f612964b03afcf0749bea/moveit_core/kinematic_constraints/src/kinematic_constraint.cpp

    Returns
    -------
    Whether the current_pose is within the specified tolerances of the
    target_pose.
    """
    # Ensure both poses are in the same frame
    assert current_pose.header.frame_id == target_pose.header.frame_id, (
        f"current_pose.header.frame_id ({current_pose.header.frame_id}) "
        f"!= target_pose.header.frame_id ({target_pose.header.frame_id})"
    )

    # Check the linear distance between the poses
    linear_displacement = ros2_numpy.numpify(
        target_pose.pose.position
    ) - ros2_numpy.numpify(current_pose.pose.position)

    linear_distance = np.linalg.norm(linear_displacement)
    if linear_distance > tolerance_position:
        return False

    # Check the angular distance between the poses
    tol = np.fabs(np.array(tolerance_orientation) * np.ones(3))
    current_orientation = R.from_quat(ros2_numpy.numpify(current_pose.pose.orientation))
    target_orientation = R.from_quat(ros2_numpy.numpify(target_pose.pose.orientation))
    diff = R.from_matrix(
        np.dot(current_orientation.as_matrix().T, target_orientation.as_matrix())
    )
    if np.any(np.fabs(diff.as_rotvec()) > tol):
        return False

    # If we made it here, then the current_pose is within the specified
    # tolerances of the target_pose.
    return True


def servo_until_pose(
    name: str,
    ns: str,
    target_pose_stamped_key: BlackboardKey,
    end_effector_frame: str = "forkTip",
    tolerance_position: float = 0.005,
    tolerance_orientation: Union[float, Tuple[float, float, float]] = 0.09,
    relaxed_tolerance_position: float = 0.005,
    relaxed_tolerance_orientation: Union[float, Tuple[float, float, float]] = 0.09,
    twist_blackboard_key: BlackboardKey = BlackboardKey("twist"),
    duration: Union[Duration, float] = 10.0,
    speed: Union[Callable[[PoseStamped], Tuple[float, float]], Tuple[float, float]] = (
        0.1,
        0.3,
    ),
    round_decimals: Optional[int] = 3,
    base_link: str = "j2n6s200_link_base",
    ignore_orientation: bool = False,
    subscribe_to_servo_status: bool = False,
    pub_topic: str = "~/servo_twist_cmds",
    f_mag_threshold: Optional[Union[BlackboardKey, float]] = None,
    t_mag_threshold: Optional[Union[BlackboardKey, float]] = None,
    viz: bool = False,
) -> py_trees.behaviour.Behaviour:
    """
    Servos until the end_effector_frame reaches within the specified tolerances
    of the target_pose_stamped.

    Parameters
    ----------
    name: The name of the behavior.
    ns: The namespace of the action server.
    end_effector_frame: The name of the end effector frame. Must be the same
        frame as the one specified in the servo config file.
    target_pose_stamped_key: The key to the target pose stamped on the blackboard.
    tolerance_position: The tolerance on the position. The robot will keep servoing
        until it reaches this distance from the target.
    tolerance_orientation: The tolerance on the orientation. The robot will keep
        servoing until it reaches this distance from the target. If a float, the
        tolerance is the same for all axes. This computes orientation tolerance
        in the same way MoveIt2 does for the "rotation vector" parameterization:
        https://github.com/ros-planning/moveit2/blob/f9989626491ca63e9f7f612964b03afcf0749bea/moveit_core/kinematic_constraints/src/kinematic_constraint.cpp
    relaxed_tolerance_position: If servoing ends in failure (e.g., at singularity,
        near collision, timeout), return success if it is this far from the target.
        The reason for this is that although a particular tolerance is ideal,
        perhaps a slightly larger tolerance is acceptable.
    relaxed_tolerance_orientation: If servoing ends in failure (e.g., at singularity,
        near collision, timeout), return success if it is this far from the target.
        The reason for this is that although a particular tolerance is ideal,
        perhaps a slightly larger tolerance is acceptable.
    twist_blackboard_key: The blackboard key for the twist.
    duration: How long to servo for. If negative, then run forever until the
        condition behavior returns SUCCESS.
    speed: Either a tuple representing the linear (m/s) and angular (rad/s)
        speed of the end effector, or a function that maps from displacement to
        the target end effector pose to the linear and angular speed.
    round_decimals: The number of decimals to round the twist to. If None, then
        the twist is not rounded.
    base_link: The name of the base link.
    ignore_orientation: if True, only servo until the position is reached, ignoring
        orientation.
    subscribe_to_servo_status: if True, subscribe to the servo status topic and
        fail on failure statuses. If False, ignore that topic.
    pub_topic: The topic to publish twist commands to.
    f_mag_threshold: If not None, then subscribe to the FT sensor and fail if the
        threshold is exceeded. If None, then do not subscribe to the FT sensor.
    t_mag_threshold: If not None, then subscribe to the FT sensor and fail if the
        threshold is exceeded. If None, then do not subscribe to the FT sensor.
    viz: if True, publish the target pose as a static TF transform.

    Returns
    -------
    A behavior that runs the servo_move behavior until either the end_effector_frame
    reaches within the specified tolerances of the target_pose_stamped or servo reaches
    near collision, on collision, or on singularity, or the duration is reached.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # This is intended to be a flexible idiom.

    # Get the absolute keys necessary for non-BlackboardBehaviors
    ee_to_target_pose_stamped_absolute_key = Blackboard.separator.join(
        [ns, "ee_to_target_pose_stamped"]
    )
    num_ticks_absolute_key = Blackboard.separator.join([ns, "num_ticks"])
    start_time_absolute_key = Blackboard.separator.join([ns, "start_time"])

    servo_inputs = {
        "twist": twist_blackboard_key,
        "duration": duration,  # timeout for Servo
        "status_on_timeout": py_trees.common.Status.FAILURE,
        "pub_topic": pub_topic,
    }
    if not subscribe_to_servo_status:
        servo_inputs["servo_status_sub_topic"] = None

    # Sense the distance from the end effector to the target pose
    sense_behavior = py_trees.composites.Sequence(
        name=f"{name} ServoUntilPose Condition",
        memory=True,
        children=[
            # Update the timestamp of the target_pose_stamped
            UpdateTimestamp(
                name=f"{name} Update Timestamp",
                ns=ns,
                inputs={
                    "stamped_msg": target_pose_stamped_key,
                    "timestamp": Time(seconds=0.0),  # Get latest transform
                },
                outputs={
                    "stamped_msg": target_pose_stamped_key,
                },
            ),
            # Get the target_pose_stamped in the end effector frame
            ApplyTransform(
                name=f"{name} ServoUntilPose GetEEDisplacement",
                ns=ns,
                inputs={
                    "stamped_msg": target_pose_stamped_key,
                    "target_frame": end_effector_frame,
                },
                outputs={
                    "transformed_msg": BlackboardKey("ee_to_target_pose_stamped"),
                },
            ),
        ],
    )
    if viz:
        # Cache the target pose on the static TF tree
        sense_behavior.add_child(
            SetStaticTransform(
                name=f"{name} ServoUntilPose PublishPose",
                ns=ns,
                inputs={
                    "transform": BlackboardKey("ee_to_target_pose_stamped"),
                    "child_frame_id": "servo_until_pose_target",
                },
            ),
        )

    check_children = []
    # if f_mag_threshold is not None or t_mag_threshold is not None:
    #     # Check the F/T threshold.
    #     kwargs = {
    #         "name": f"{name} CheckFT",
    #         "ns": ns,
    #     }
    #     if f_mag_threshold is not None:
    #         kwargs["f_mag"] = f_mag_threshold
    #     if t_mag_threshold is not None:
    #         kwargs["t_mag"] = t_mag_threshold
    #     # NOTE: Using `ft_thresh_satisfied` might be inefficient if the tree this
    #     # idiom is used in has other behaviors that subscribe to the F/T threshold.
    #     check_children.append(
    #         py_trees.decorators.Inverter(
    #             name=f"{name} ServoUntilPose CheckFT Inverter",
    #             child=ft_thresh_satisfied(**kwargs),
    #         )
    #     )

    # Check whether the ee_to_target_pose_stamped is within the tolerances.
    # Note that this behavior can technically return FAILURE for a reason
    # other than the condition not being met, specifically if the blackboard
    # variable doens't exist or the permissions are not correctly set.
    # Therefore, it is crucial to be vigilant when testing this idiom.
    check_children.append(
        py_trees.behaviours.CheckBlackboardVariableValue(
            name=f"{name} ServoUntilPose CheckTolerances",
            check=py_trees.common.ComparisonExpression(
                variable=ee_to_target_pose_stamped_absolute_key,
                value=PoseStamped(
                    header=Header(frame_id=end_effector_frame),
                    pose=Pose(
                        position=Point(x=0.0, y=0.0, z=0.0),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                ),
                operator=partial(
                    pose_within_tolerances,
                    tolerance_position=tolerance_position,
                    tolerance_orientation=np.inf
                    if ignore_orientation
                    else tolerance_orientation,
                ),
            ),
        )
    )
    check_behavior = py_trees.composites.Sequence(
        name=f"{name} ServoUntilPose CheckSequence",
        memory=True,
        children=check_children,
    )

    # Compute the twist based on the ee_to_target_pose_stamped
    compute_twist_behavior = py_trees.composites.Sequence(
        name=f"{name} ServoUntilPose Condition",
        memory=False,
        children=[
            # Monitor the rate at which this idiom is running
            TrackHz(
                name=f"{name} TrackHz",
                ns=ns,
                inputs={
                    "num_ticks": BlackboardKey("num_ticks"),
                    "start_time": BlackboardKey("start_time"),
                },
                outputs={
                    "hz": BlackboardKey("servoHz"),
                    "num_ticks": BlackboardKey("num_ticks"),
                    "start_time": BlackboardKey("start_time"),
                },
            ),
            # Compute the twist
            PoseStampedToTwistStamped(
                name=f"{name} {SERVO_UNTIL_POSE_DISTANCE_BEHAVIOR_NAME}",
                ns=ns,
                inputs={
                    "pose_stamped": BlackboardKey("ee_to_target_pose_stamped"),
                    "speed": speed,
                    "hz": BlackboardKey("servoHz"),
                    "round_decimals": round_decimals,
                    "angular_override": Vector3(x=0.0, y=0.0, z=0.0)
                    if ignore_orientation
                    else None,
                },
                outputs={
                    "twist_stamped": BlackboardKey("twist_in_ee_frame"),
                },
            ),
            # Convert to base frame
            ApplyTransform(
                name=f"{name} ServoUntilPose GetEETwist",
                ns=ns,
                inputs={
                    "stamped_msg": BlackboardKey("twist_in_ee_frame"),
                    "target_frame": base_link,
                },
                outputs={
                    "transformed_msg": twist_blackboard_key,
                },
            ),
        ],
    )

    return py_trees.composites.Selector(
        name=name,
        memory=True,
        children=[
            py_trees.composites.Sequence(
                name=f"{name} ServoUntilPose",
                memory=True,
                children=[
                    # Unset num_ticks and start_time so TrackHz only tracks the rate
                    # of this servo_until idiom, irrespective of prior times this idiom
                    # may have been called (e.g., other times this tree was ticked).
                    UnsetBlackboardVariable(
                        name=f"{name} ServoUntilPose UnsetBlackboardVariable num_ticks",
                        key=num_ticks_absolute_key,
                    ),
                    UnsetBlackboardVariable(
                        name=f"{name} ServoUntilPose UnsetBlackboardVariable num_ticks",
                        key=start_time_absolute_key,
                    ),
                    servo_until(
                        name=name,
                        ns=ns,
                        sense=sense_behavior,
                        check=check_behavior,
                        compute_twist=compute_twist_behavior,
                        servo_inputs=servo_inputs,
                        f_mag_threshold=f_mag_threshold,
                        t_mag_threshold=t_mag_threshold,
                    ),
                ],
            ),
            py_trees.behaviours.CheckBlackboardVariableValue(
                name=f"{name} Final CheckTolerances",
                check=py_trees.common.ComparisonExpression(
                    variable=ee_to_target_pose_stamped_absolute_key,
                    value=PoseStamped(
                        header=Header(frame_id=end_effector_frame),
                        pose=Pose(
                            position=Point(x=0.0, y=0.0, z=0.0),
                            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                        ),
                    ),
                    operator=partial(
                        pose_within_tolerances,
                        tolerance_position=relaxed_tolerance_position,
                        tolerance_orientation=np.inf
                        if ignore_orientation
                        else relaxed_tolerance_orientation,
                    ),
                ),
            ),
        ],
    )
