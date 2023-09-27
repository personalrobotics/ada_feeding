#!/usr/bin/env python3
"""
This module defines unit tests for the eventually_swiss idiom.
"""

# Standard imports
import time
from typing import List, Optional, Union

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard


class TickCounterWithTerminateTimestamp(py_trees.behaviours.TickCounter):
    """
    This class is identical to TickCounter, except that it also stores the
    timestamp when the behavior terminated.
    """

    def __init__(
        self,
        name: str,
        duration: int,
        completion_status: py_trees.common.Status,
        ns: str = "/",
    ):
        """
        Initialise the behavior.
        """
        super().__init__(
            name=name, duration=duration, completion_status=completion_status
        )
        self.termination_new_status = None
        self.termination_timestamp = None
        self.num_times_ticked_to_non_running_status = 0

        # Create a blackboard client to store this behavior's status,
        # counter, termination_new_status, and termination_timestamp.
        self.blackboard = self.attach_blackboard_client(name=name, namespace=ns)
        self.blackboard.register_key(key="status", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="counter", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key="num_times_ticked_to_non_running_status",
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key="termination_new_status", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="termination_timestamp", access=py_trees.common.Access.WRITE
        )

        # Initialize the blackboard
        self.blackboard.status = py_trees.common.Status.INVALID
        self.blackboard.counter = 0
        self.blackboard.termination_new_status = self.termination_new_status
        self.blackboard.termination_timestamp = self.termination_timestamp

    def initialise(self) -> None:
        """Reset the tick counter."""
        self.counter = 0

        # Update the blackboard.
        self.blackboard.status = self.status
        self.blackboard.counter = self.counter
        self.blackboard.termination_new_status = self.termination_new_status
        self.blackboard.termination_timestamp = self.termination_timestamp

    def update(self) -> py_trees.common.Status:
        """
        Update the behavior.
        """
        new_status = super().update()

        if new_status != py_trees.common.Status.RUNNING:
            self.num_times_ticked_to_non_running_status += 1

        # Update the blackboard.
        self.blackboard.status = new_status
        self.blackboard.counter = self.counter
        self.blackboard.num_times_ticked_to_non_running_status = (
            self.num_times_ticked_to_non_running_status
        )

        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Terminate the behavior.
        """
        self.termination_new_status = new_status
        self.termination_timestamp = time.time()

        # Update the blackboard.
        self.blackboard.termination_new_status = self.termination_new_status
        self.blackboard.termination_timestamp = self.termination_timestamp
        # Although self.status will be set in the `stop` method that called
        # this, it won't set on the blackboard. So we set that here.
        self.blackboard.status = new_status


def check_count_status(
    behaviors: List[Union[TickCounterWithTerminateTimestamp, str]],
    counts: List[int],
    statuses: List[py_trees.common.Status],
    num_times_ticked_to_non_running_statuses: List[int],
    descriptor: str = "",
) -> None:
    """
    Takes in a list of TickCounter behaviors and checks that their counts and
    statuses are correct.

    Parameters
    ----------
    behaviors: The list of behaviors to check. The values are either behaviors,
        in which case the attributes of the behavior will directly be checked,
        or strings, which is the blackboard namespace where the behavior has
        stored its attributes.
    counts: The expected counts for each behavior.
    statuses: The expected statuses for each behavior.
    num_times_ticked_to_non_running_statuses: The expected number of times each
        behavior had a tick resulting in a non-running status.
    """
    assert (
        len(behaviors) == len(counts) == len(statuses)
    ), "lengths of behaviors, counts, and statuses must be equal"

    for i, behavior in enumerate(behaviors):
        # Get the actual count and status
        if isinstance(behavior, str):
            name = behavior
            actual_count = Blackboard().get(
                Blackboard.separator.join([behavior, "counter"])
            )
            actual_status = Blackboard().get(
                Blackboard.separator.join([behavior, "status"])
            )
            actual_num_times_ticked_to_non_running_statuses = Blackboard().get(
                Blackboard.separator.join(
                    [behavior, "num_times_ticked_to_non_running_status"]
                )
            )
        else:
            name = behavior.name
            actual_count = behavior.counter
            actual_status = behavior.status
            actual_num_times_ticked_to_non_running_statuses = (
                behavior.num_times_ticked_to_non_running_status
            )

        # Check the actual count and status against the expected ones
        assert actual_count == counts[i], (
            f"behavior '{name}' actual count {actual_count}, "
            f"expected count {counts[i]}, "
            f"{descriptor}"
        )
        assert actual_status == statuses[i], (
            f"behavior '{name}' actual status {actual_status}, "
            f"expected status {statuses[i]}, "
            f"{descriptor}"
        )
        assert (
            actual_num_times_ticked_to_non_running_statuses
            == num_times_ticked_to_non_running_statuses[i]
        ), (
            f"behavior '{name}' actual num_times_ticked_to_non_running_statuses "
            f"{actual_num_times_ticked_to_non_running_statuses}, "
            f"expected num_times_ticked_to_non_running_statuses "
            f"{num_times_ticked_to_non_running_statuses[i]}, "
            f"{descriptor}"
        )


def check_termination_new_statuses(
    behaviors: List[Union[TickCounterWithTerminateTimestamp, str]],
    statuses: List[Optional[py_trees.common.Status]],
    descriptor: str = "",
) -> None:
    """
    Checkes that `terminate` either has not been called on the behavior, or
    that it has been called with the correct new status.

    Parameters
    ----------
    behaviors: The list of behaviors to check. The values are either behaviors,
        in which case the attributes of the behavior will directly be checked,
        or strings, which is the blackboard namespace where the behavior has
        stored its attributes.
    statuses: The expected new statuses for each behavior when `terminate` was
        called, or `None` if `terminate` was not expected to be called.
    """
    assert len(behaviors) == len(
        statuses
    ), "lengths of behaviors and statuses must be equal"

    for i, behavior in enumerate(behaviors):
        # Get the actual termination_new_status
        if isinstance(behavior, str):
            name = behavior
            actual_termination_new_status = Blackboard().get(
                Blackboard.separator.join([behavior, "termination_new_status"])
            )
        else:
            name = behavior.name
            actual_termination_new_status = behavior.termination_new_status

        # Check the actual termination_new_status against the expected one
        if statuses[i] is None:
            assert actual_termination_new_status is None, (
                f"behavior '{name}' expected termination_new_status None, actual "
                f"termination_new_status {actual_termination_new_status}, "
                f"{descriptor}"
            )
        else:
            assert actual_termination_new_status == statuses[i], (
                f"behavior '{name}' actual termination_new_status "
                f"{actual_termination_new_status}, expected termination_new_status "
                f"{statuses[i]}, {descriptor}"
            )


def check_termination_order(
    behaviors: List[Union[TickCounterWithTerminateTimestamp, str]],
    descriptor: str = "",
) -> None:
    """
    Checks that the behaviors terminated in the correct order.

    Parameters
    ----------
    behaviors: The list of behaviors to check, in the order that `terminate`
        should have been called on them. The values are either behaviors, in
        which case the attributes of the behavior will directly be checked, or
        strings, which is the blackboard namespace where the behavior has stored
        its attributes.
    """
    for i in range(len(behaviors) - 1):
        # Get the actual termination_timestamp
        if isinstance(behaviors[i], str):
            curr_name = behaviors[i]
            actual_curr_termination_timestamp = Blackboard().get(
                Blackboard.separator.join([behaviors[i], "termination_timestamp"])
            )
        else:
            curr_name = behaviors[i].name
            actual_curr_termination_timestamp = behaviors[i].termination_timestamp

        if isinstance(behaviors[i + 1], str):
            next_name = behaviors[i + 1]
            actual_next_termination_timestamp = Blackboard().get(
                Blackboard.separator.join([behaviors[i + 1], "termination_timestamp"])
            )
        else:
            next_name = behaviors[i + 1].name
            actual_next_termination_timestamp = behaviors[i + 1].termination_timestamp

        # Check the actual termination_timestamp against the expected one
        assert actual_curr_termination_timestamp <= actual_next_termination_timestamp, (
            f"behavior '{curr_name}' terminated after behavior "
            f"'{next_name}', when it should have terminated before, "
            f"{descriptor}"
        )
