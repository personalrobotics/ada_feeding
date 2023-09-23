#!/usr/bin/env python3
"""
This module defines unit tests for the eventually_swiss idiom.
"""

# Standard imports
from enum import Enum
from functools import partial
from typing import List

# Third-party imports
import py_trees

# Local imports
from ada_feeding.idioms import eventually_swiss
from .helpers import (
    TickCounterWithTerminateTimestamp,
    check_count_status,
    check_termination_new_statuses,
    check_termination_order,
)

# pylint: disable=duplicate-code
# `test_scoped_behavior` and `test_eventually_swiss` have similar code because
# they are similar idioms. That is okay.
# pylint: disable=redefined-outer-name
# When generating tests, we use global variables with the same names as
# variables in the functions. That is okay, since the functions don't need
# access to the global variables.


class ExecutionCase(Enum):
    """
    Tree execution can broadly fall into one of the below three cases.
    """

    NONE = 0
    WORKER_RUNNING = 1
    WORKER_TERMINATED_CALLBACK_RUNNING = 2
    TREE_TERMINATED = 3


def generate_test(
    worker_duration: int,
    worker_completion_status: py_trees.common.Status,
    on_success_duration: int,
    on_success_completion_status: py_trees.common.Status,
    on_failure_duration: int,
    on_failure_completion_status: py_trees.common.Status,
    on_preempt_duration: int,
    on_preempt_completion_status: py_trees.common.Status,
):
    """
    Generates a worker, on_success, on_failure, and on_preempt behavior with the
    specified durations and completion statuses.

    Note that this always generates the multi-tick version of eventually_swiss,
    where it returns `on_success` status but not `on_failure` status.
    """
    # pylint: disable=too-many-arguments
    # Necessary to create a versatile test generation function.

    # Setup the test
    worker = TickCounterWithTerminateTimestamp(
        name="Worker",
        duration=worker_duration,
        completion_status=worker_completion_status,
        ns="/worker",
    )
    on_success = TickCounterWithTerminateTimestamp(
        name="On Success",
        duration=on_success_duration,
        completion_status=on_success_completion_status,
        ns="/on_success",
    )
    on_failure = TickCounterWithTerminateTimestamp(
        name="On Failure",
        duration=on_failure_duration,
        completion_status=on_failure_completion_status,
        ns="/on_failure",
    )
    on_preempt = TickCounterWithTerminateTimestamp(
        name="On Preempt",
        duration=on_preempt_duration,
        completion_status=on_preempt_completion_status,
        ns="/on_preempt",
    )
    root = eventually_swiss(
        name="Eventually Swiss",
        workers=[worker],
        on_success=on_success,
        on_failure=on_failure,
        on_preempt=on_preempt,
        on_preempt_single_tick=False,
    )
    return root, worker, on_success, on_failure, on_preempt


def combined_test(
    worker_completion_status: py_trees.common.Status,
    callback_completion_status: py_trees.common.Status,
    global_num_cycles: int = 2,
    preempt_times: List[ExecutionCase] = [ExecutionCase.NONE, ExecutionCase.NONE],
) -> None:
    """
    This function ticks the root to completion `global_num_cycles` times and checks
    the following three cases:

    Case WORKER_RUNNING:
     - While the worker is RUNNING, `on_success`, `on_failure`, and `on_preempt`
       should not be ticked, none of the functions should be terminated, and the
       root should be running.

    When the worker succeeds:
        Case WORKER_TERMINATED_CALLBACK_RUNNING:
         - While `on_success` is RUNNING, `on_failure` and `on_preempt` should
           not be ticked, none of the functions should be terminated, and the
           root should be running.
        Case TREE_TERMINATED:
         - When `on_success` returns `callback_completion_status`, `on_failure`
           and `on_preempt` should not be ticked, none of the functions should
           be terminated, and the root should return `callback_completion_status`.

    When the worker fails:
        Case WORKER_TERMINATED_CALLBACK_RUNNING:
         - While `on_failure` is RUNNING, `on_success` and `on_preempt` should
           not be ticked, none of the functions should be terminated, and the
           root should be running.
        Case TREE_TERMINATED:
         - When `on_failure` returns `callback_completion_status`, `on_success`
           and `on_preempt` should not be ticked, none of the functions should
           be terminated, and the root should return FAILURE.

    Additionally, this function can terminate the tree up to once per cycle.
    For cycle i, this function will terminate the tree depending on the value of
    `preempt_times[i]`:
      - None: Don't terminate this cycle.
      - WORKER_RUNNING: Terminate the tree after the first tick when the worker
        is RUNNING.
      - WORKER_TERMINATED_CALLBACK_RUNNING: Terminate the tree after the first
        tick when the worker has terminated and the callback is RUNNING.
      - TREE_TERMINATED: Terminate the tree after the tick
        when the worker has terminated and the callback has terminated (i.e., after
        the tick where the root returns a non-RUNNING status).
    After terminating the tree, in the first two cases, this function checks that
    the tree has not ticked `worker`, `on_success`, or `on_failure` any more, but
    has ticked `on_preempt` to completion. It also checks that the tree has
    terminated in the correct order: `worker` -> `on_success`/`on_failure` -> `on_preempt`.
    In the third case, since the tree has already reached a non-RUNNING status,
    `on_preempt` should not be run, and this function verifies that.

    Parameters
    ----------
    worker_completion_status: The completion status of the worker.
    callback_completion_status: The completion status of the callback.
    global_num_cycles: The number of times to tick the root to completion.
    preempt_times: A list of ExecutionCase values, one for each cycle. If None,
        don't preempt the tree during that cycle.
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # pylint: disable=too-many-nested-blocks
    # This is where the bulk of the work to test eventually_swiss is done, so
    # it's hard to reduce the number of locals, branches, and statements.
    # pylint: disable=dangerous-default-value
    # A default value of a list is fine in this case.

    assert len(preempt_times) >= global_num_cycles, "Malformed test case."

    # Setup the test
    worker_duration = 2
    callback_duration = 3
    other_callbacks_completion_status = py_trees.common.Status.SUCCESS
    root, worker, on_success, on_failure, on_preempt = generate_test(
        worker_duration=worker_duration,
        worker_completion_status=worker_completion_status,
        on_success_duration=callback_duration,
        on_success_completion_status=(
            callback_completion_status
            if worker_completion_status == py_trees.common.Status.SUCCESS
            else other_callbacks_completion_status
        ),
        on_failure_duration=callback_duration,
        on_failure_completion_status=(
            callback_completion_status
            if worker_completion_status == py_trees.common.Status.FAILURE
            else other_callbacks_completion_status
        ),
        on_preempt_duration=callback_duration,
        on_preempt_completion_status=other_callbacks_completion_status,
    )

    # Get the number of ticks it should take to terminate this tree.
    num_ticks_to_terminate = worker_duration + callback_duration + 1

    # Initialize the expected counts, statuses, termination_new_statuses, and
    # root status for the tests
    behaviors = [worker, on_success, on_failure, on_preempt]
    expected_counts = [0, 0, 0, 0]
    expected_statuses = [
        py_trees.common.Status.INVALID,
        py_trees.common.Status.INVALID,
        py_trees.common.Status.INVALID,
        py_trees.common.Status.INVALID,
    ]
    expected_termination_new_statuses = [None, None, None, None]

    # Tick the tree
    for num_cycles in range(global_num_cycles):
        # The worker's count gets re-initialized at the beginning of every cycle.
        expected_counts[0] = 0
        # The worker, on_success, and on_failure get reset to INVALID at the
        # beginning of every cycle.
        if num_cycles > 0:
            expected_statuses[0] = py_trees.common.Status.INVALID
            expected_termination_new_statuses[0] = py_trees.common.Status.INVALID
            if (
                worker_completion_status == py_trees.common.Status.SUCCESS
                and expected_statuses[1] != py_trees.common.Status.INVALID
            ):
                expected_termination_new_statuses[1] = py_trees.common.Status.INVALID
                expected_statuses[1] = py_trees.common.Status.INVALID
            if (
                worker_completion_status == py_trees.common.Status.FAILURE
                and expected_statuses[2] != py_trees.common.Status.INVALID
            ):
                expected_termination_new_statuses[2] = py_trees.common.Status.INVALID
                expected_statuses[2] = py_trees.common.Status.INVALID
        for num_ticks in range(1, num_ticks_to_terminate + 1):
            descriptor = f"num_ticks {num_ticks}, num_cycles {num_cycles}"
            # Tick the tree
            root.tick_once()
            # Get the expected counts, statuses, termination_new_statuses, and
            # root status.
            # The worker is still running. WORKER_RUNNING case.
            if num_ticks <= worker_duration:
                execution_case = ExecutionCase.WORKER_RUNNING
                expected_counts[0] += 1  # The worker should have gotten ticked.
                expected_statuses[0] = py_trees.common.Status.RUNNING
                root_expected_status = py_trees.common.Status.RUNNING
            # The worker has terminated, but the success/failure callback is still running.
            # WORKER_TERMINATED_CALLBACK_RUNNING case.
            elif num_ticks <= worker_duration + callback_duration:
                execution_case = ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING
                if num_ticks == worker_duration + 1:
                    # The worker terminates on the first tick after `worker_duration`
                    expected_counts[0] += 1
                    # on_success and on_failure only gets reinitialized after the
                    # worker terminates.
                    expected_counts[1] = 0
                    expected_counts[2] = 0
                    # The worker status gets set
                    expected_statuses[0] = worker_completion_status
                    expected_termination_new_statuses[0] = worker_completion_status
                elif worker_completion_status == py_trees.common.Status.FAILURE:
                    # The Selector with memory unnecessarily sets previous children to
                    # INVALID the tick after they fail, hence the below switch.
                    # https://github.com/splintered-reality/py_trees/blob/0d5b39f2f6333c504406d8a63052c456c6bd1ce5/py_trees/composites.py#L427
                    expected_statuses[0] = py_trees.common.Status.INVALID
                    expected_termination_new_statuses[
                        0
                    ] = py_trees.common.Status.INVALID
                if worker_completion_status == py_trees.common.Status.SUCCESS:
                    expected_counts[1] += 1
                    expected_statuses[1] = py_trees.common.Status.RUNNING
                elif worker_completion_status == py_trees.common.Status.FAILURE:
                    expected_counts[2] += 1
                    expected_statuses[2] = py_trees.common.Status.RUNNING
                else:
                    assert (
                        False
                    ), f"Unexpected worker_completion_status {worker_completion_status}."
                root_expected_status = py_trees.common.Status.RUNNING
            # The success/failure callback has terminated.
            # TREE_TERMINATED case.
            elif num_ticks == num_ticks_to_terminate:
                execution_case = ExecutionCase.TREE_TERMINATED
                if worker_completion_status == py_trees.common.Status.SUCCESS:
                    expected_counts[1] += 1
                    expected_statuses[1] = callback_completion_status
                    expected_termination_new_statuses[1] = callback_completion_status
                elif worker_completion_status == py_trees.common.Status.FAILURE:
                    expected_counts[2] += 1
                    expected_statuses[2] = callback_completion_status
                    expected_termination_new_statuses[2] = callback_completion_status
                else:
                    assert (
                        False
                    ), f"Unexpected worker_completion_status {worker_completion_status}."
                root_expected_status = (
                    callback_completion_status
                    if worker_completion_status == py_trees.common.Status.SUCCESS
                    else py_trees.common.Status.FAILURE
                )
            else:
                assert False, (
                    f"Should not get here, num_ticks {num_ticks}, "
                    f"num_cycles {num_cycles}"
                )

            # Run the tests
            check_count_status(
                behaviors=behaviors,
                counts=expected_counts,
                statuses=expected_statuses,
                descriptor=descriptor,
            )
            check_termination_new_statuses(
                behaviors=behaviors,
                statuses=expected_termination_new_statuses,
                descriptor=descriptor,
            )
            assert (
                root.status == root_expected_status
            ), f"root status {root.status} is not {root_expected_status}, {descriptor}"

            # Preempt if requested
            if preempt_times[num_cycles] == execution_case:
                root.stop(py_trees.common.Status.INVALID)
                descriptor += " after preemption"

                # Update the expected termination of all behaviors but `on_preempt`
                termination_order_on_success = []
                termination_order_on_failure = []
                for i in range(3):
                    if expected_statuses[i] != py_trees.common.Status.INVALID:
                        expected_statuses[i] = py_trees.common.Status.INVALID
                        expected_termination_new_statuses[
                            i
                        ] = py_trees.common.Status.INVALID
                        if i == 1:
                            termination_order_on_success.append(behaviors[i])
                        elif i == 2:
                            termination_order_on_failure.append(behaviors[i])
                        else:
                            termination_order_on_success.append(behaviors[i])
                            termination_order_on_failure.append(behaviors[i])
                root_expected_status = py_trees.common.Status.INVALID
                # `on_preempt` should only get ticked if the worker/callback
                # have not yet terminated. If they have terminated, the root
                # is considered complete and there is no reason to run `on_preempt`.
                if execution_case in [
                    ExecutionCase.WORKER_RUNNING,
                    ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
                ]:
                    # `on_preempt` should get ticked to completion
                    expected_counts[3] = callback_duration + 1

                    # Because `on_preempt` is not officially a part of the tree,
                    # it won't get called as part of the preemption. So it's
                    # status will be its terminal status.
                    expected_statuses[3] = other_callbacks_completion_status
                    expected_termination_new_statuses[
                        3
                    ] = other_callbacks_completion_status
                    termination_order_on_success.append(behaviors[3])
                    termination_order_on_failure.append(behaviors[3])

                # Run the preemption tests
                check_count_status(
                    behaviors=behaviors,
                    counts=expected_counts,
                    statuses=expected_statuses,
                    descriptor=descriptor,
                )
                check_termination_new_statuses(
                    behaviors=behaviors,
                    statuses=expected_termination_new_statuses,
                    descriptor=descriptor,
                )
                check_termination_order(termination_order_on_success, descriptor)
                check_termination_order(termination_order_on_failure, descriptor)
                assert (
                    root.status == root_expected_status
                ), f"root status {root.status} is not {root_expected_status}, {descriptor}"

                # End this cycle
                break


################################################################################
# Generate all tests with 2 cycles
################################################################################

for worker_completion_status in [
    py_trees.common.Status.SUCCESS,
    py_trees.common.Status.FAILURE,
]:
    for callback_completion_status in [
        py_trees.common.Status.SUCCESS,
        py_trees.common.Status.FAILURE,
    ]:
        for first_preempt in [
            ExecutionCase.NONE,
            ExecutionCase.WORKER_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
            ExecutionCase.TREE_TERMINATED,
        ]:
            for second_preempt in [
                ExecutionCase.NONE,
                ExecutionCase.WORKER_RUNNING,
                ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
                ExecutionCase.TREE_TERMINATED,
            ]:
                test_name = (
                    f"test_worker_{worker_completion_status.name}_callback_"
                    f"{callback_completion_status.name}_first_preempt_{first_preempt.name}_"
                    f"second_preempt_{second_preempt.name}"
                )
                globals()[test_name] = partial(
                    combined_test,
                    worker_completion_status=worker_completion_status,
                    callback_completion_status=callback_completion_status,
                    preempt_times=[first_preempt, second_preempt],
                )
