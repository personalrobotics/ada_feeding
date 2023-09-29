#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveIt2Excute behavior, which uses pymoveit2
to execute and JointTrajectory.
"""

# Standard imports
from typing import Union, Optional

# Third-party imports
from action_msgs.msg import GoalStatus
from overrides import override
from moveit_msgs.msg import MoveItErrorCodes
import py_trees
from pymoveit2 import MoveIt2State
from trajectory_msgs.msg import JointTrajectory

# Local imports
from ada_feeding.helpers import BlackboardKey, get_moveit2_object
from ada_feeding.behaviors import BlackboardBehavior


class MoveIt2Execute(BlackboardBehavior):
    """
    Runs moveit2.py execute with the provided
    JointTrajectory.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self, trajectory: Union[BlackboardKey, Optional[JointTrajectory]]
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        trajectory: JointTrajectory to execute. If None, return SUCCESS immediately.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        error_code: Optional[
            BlackboardKey
        ],  # Union[MoveItErrorCodes, GoalStatus] == int
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        error_code: specifies the manner of execution (or goal) failure
                    See actionlib_msgs/GoalStatus and moveit_msgs/MoveItErrorCodes for more info
                    <0: MoveItErrorCodes Failure
                    0, 3: Never Returned (GoalStatus PENDING, SUCCEEDED)
                    1: MoveItErrorCodes Success (GoalStatus ACTIVE never returned)
                    2, >4: GoalStatus Failure
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Get Node from Kwargs
        self.node = kwargs["node"]

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            self.node,
        )

    @override
    def initialise(self) -> None:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        self.motion_future = None

        # First tick: send the trajectory to MoveIt
        # Note: this *could* block, but it is unlikely
        with self.moveit2_lock:
            if (
                self.blackboard_exists("trajectory")
                and self.blackboard_get("trajectory") is not None
            ):
                self.moveit2.execute(self.blackboard_get("trajectory"))

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init, too-many-return-statements
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Handle empty trajectory
        if (
            not self.blackboard_exists("trajectory")
            or self.blackboard_get("trajectory") is None
        ):
            self.blackboard_set("error_code", MoveItErrorCodes.SUCCESS)
            return py_trees.common.Status.SUCCESS

        # Lock MoveIt2 Object
        if self.moveit2_lock.locked():
            return py_trees.common.Status.RUNNING
        with self.moveit2_lock:
            if self.motion_future is None:
                query_state = self.moveit2.query_state()
                if query_state == MoveIt2State.REQUESTING:
                    # The goal has been sent to the action server, but not yet accepted
                    return py_trees.common.Status.RUNNING
                if query_state == MoveIt2State.EXECUTING:
                    # The goal has been accepted and is executing. In this case
                    # don't return a status since we drop down into the below
                    # for when the robot is in motion.
                    self.motion_future = self.moveit2.get_execution_future()
                elif query_state == MoveIt2State.IDLE:
                    last_error_code = self.moveit2.get_last_execution_error_code()
                    self.blackboard_set("error_code", last_error_code)
                    if last_error_code is None or last_error_code.val != 1:
                        # If we get here then something went wrong (e.g., controller
                        # is already executing a trajectory, action server not
                        # available, goal was rejected, etc.)
                        self.logger.error(
                            f"{self.name} [MoveTo::update()] Failed to execute trajectory before "
                            "goal was accepted!"
                        )
                        return py_trees.common.Status.FAILURE
                    # If we get here, the goal finished executing within the
                    # last tick.
                    return py_trees.common.Status.SUCCESS

            if self.motion_future.done():
                # The goal has finished executing
                if self.motion_future.result().status == GoalStatus.STATUS_SUCCEEDED:
                    error_code = self.motion_future.result().result.error_code
                    self.blackboard_set("error_code", error_code)
                    if error_code.val == MoveItErrorCodes.SUCCESS:
                        # The goal succeeded
                        return py_trees.common.Status.SUCCESS
                    # The goal failed (execution)
                    return py_trees.common.Status.FAILURE
                # The goal failed (actionlib)
                self.blackboard_set("error_code", self.motion_future.result().status)
                return py_trees.common.Status.FAILURE

        # The goal is still executing
        return py_trees.common.Status.RUNNING
