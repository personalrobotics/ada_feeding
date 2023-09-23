#!/usr/bin/env python3
"""
This module defines unit tests for the scoped_behaviour idiom.
"""

# Standard imports
from enum import Enum
from typing import List, Optional

# Third-party imports
import py_trees

# Local imports
from ada_feeding.idioms import scoped_behavior
from .helpers import (
    TickCounterWithTerminateTimestamp,
    check_count_status,
    check_termination_new_statuses,
    check_termination_order,
)


class ExecutionCase(Enum):
    """
    Tree execution can broadly fall into one of the below three cases.
    """

    PRE_RUNNING = 0
    WORKERS_RUNNING = 1
    POST_RUNNING = 2
    TREE_TERMINATED = 3


def generate_test(
    pre_duration: int,
    pre_completion_status: py_trees.common.Status,
    worker_duration: int,
    worker_completion_status: py_trees.common.Status,
    post_duration: int,
    post_completion_status: py_trees.common.Status,
):
    """
    Generates a worker, pre, and post behavior with the
    specified durations and completion statuses.
    """
    # pylint: disable=too-many-arguments
    # Necessary to create a versatile test generation function.

    # Setup the test
    post_ns = "/post"
    pre = TickCounterWithTerminateTimestamp(
        name="Pre",
        duration=pre_duration,
        completion_status=pre_completion_status,
        ns="/pre",
    )
    worker = TickCounterWithTerminateTimestamp(
        name="Worker",
        duration=worker_duration,
        completion_status=worker_completion_status,
        ns="/worker",
    )

    def post_fn():
        """
        Return the post behavior.
        """
        return TickCounterWithTerminateTimestamp(
            name="Post",
            duration=post_duration,
            completion_status=post_completion_status,
            ns=post_ns,
        )

    root = scoped_behavior(
        name="Root",
        pre_behavior=pre,
        main_behaviors=[worker],
        post_behavior_fn=post_fn,
    )
    return root, pre, worker, post_ns


def combined_test(
    pre_completion_status: py_trees.common.Status,
    worker_completion_status: py_trees.common.Status,
    post_completion_status: py_trees.common.Status,
    global_num_cycles: int = 2,
    preempt_times: List[Optional[ExecutionCase]] = [None, None],
) -> None:
    """
    This function ticks the root to completion `global_num_cycles` times and checks
    the following three cases:

    Case PRE_RUNNING:
     - While `pre` is RUNNING, `worker` and `post` should not be ticked, none of
       the functions should be terminated, and the root should be running.

    Case WORKERS_RUNNING:
     - While `worker` is RUNNING, `post` should not be ticked, only `pre` should be
       terminated, and the root should be running.

    Case POST_RUNNING:
     - While `post` is RUNNING, only `pre` and `worker` should be terminated, and
       the root should be running.

    Case TREE_TERMINATED:
     - When the root returns a non-RUNNING status, `pre`, `worker`, and `post`
       should be terminated, and the root should return the correct status.

    Additionally, this function can terminate the tree up to once per cycle.
    For cycle i, this function will terminate the tree depending on the value of
    `preempt_times[i]`:
      - None: Don't terminate this cycle.
      - PRE_RUNNING: Terminate the tree after the first tick when the pre is
        RUNNING.
      - WORKERS_RUNNING: Terminate the tree after the first tick when the worker
        is RUNNING.
      - POST_RUNNING: Terminate the tree after the first tick when the post is
        RUNNING.
      - TREE_TERMINATED: Terminate the tree after the tick when the root returns
        a non-RUNNING status.
    After terminating the tree, in the first three cases, this function checks that
    the tree has not ticked `pre` or `worker` any more, but
    has ticked `post` to completion. It also checks that the tree has
    terminated in the correct order: `pre` -> `worker` -> `pose`.
    In the third case, since the tree has already reached a non-RUNNING status,
    nothing should change other than the statuses of the behaviors.

    Parameters
    ----------
    pre_completion_status: The completion status of the pre behavior.
    worker_completion_status: The completion status of the worker behavior.
    post_completion_status: The completion status of the post behavior.
    global_num_cycles: The number of times to tick the tree to completion.
    preempt_times: A list of ExecutionCase enums, one for each cycle.
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # pylint: disable=too-many-nested-blocks
    # This is where the bulk of the work to test eventually_swiss is done, so
    # it's hard to reduce the number of locals, branches, and statements.
    # pylint: disable=dangerous-default-value
    # A default value of a list is fine in this case.

    assert len(preempt_times) >= global_num_cycles, "Malformed test case."

    # Setup the test
    pre_duration = 3
    worker_duration = 2
    post_duration = 6
    root, pre, worker, post_ns = generate_test(
        pre_duration=pre_duration,
        pre_completion_status=pre_completion_status,
        worker_duration=worker_duration,
        worker_completion_status=worker_completion_status,
        post_duration=post_duration,
        post_completion_status=post_completion_status,
    )

    # Get the number of ticks it should take to terminate this tree.
    num_ticks_to_terminate = (
        pre_duration
        + (
            worker_duration
            if pre_completion_status == py_trees.common.Status.SUCCESS
            else 0
        )
        + post_duration
        + 1
    )

    # Initialize the expected counts, statuses, termination_new_statuses, and
    # root status for the tests
    behaviors = [pre, worker, post_ns]
    expected_counts = [0, 0, 0]
    expected_statuses = [
        py_trees.common.Status.INVALID,
        py_trees.common.Status.INVALID,
        py_trees.common.Status.INVALID,
    ]
    expected_termination_new_statuses = [None, None, None]

    # Tick the tree
    for num_cycles in range(global_num_cycles):
        # The pre's count gets re-initialized at the beginning of every cycle.
        expected_counts[0] = 0
        # The pre, worker, and post get reset to INVALID at the
        # beginning of every cycle.
        if num_cycles > 0:
            for i in range(3):
                expected_statuses[i] = py_trees.common.Status.INVALID
                if expected_termination_new_statuses[i] is not None:
                    expected_termination_new_statuses[
                        i
                    ] = py_trees.common.Status.INVALID
        for num_ticks in range(1, num_ticks_to_terminate + 1):
            descriptor = f"num_ticks {num_ticks}, num_cycles {num_cycles}"
            # Tick the tree
            root.tick_once()
            # Get the expected counts, statuses, termination_new_statuses, and
            # root status.
            # The pre is still running. PRE_RUNNING case.
            if num_ticks <= pre_duration:
                execution_case = ExecutionCase.PRE_RUNNING
                expected_counts[0] += 1  # The pre should have gotten ticked.
                expected_statuses[0] = py_trees.common.Status.RUNNING
                root_expected_status = py_trees.common.Status.RUNNING
            # The pre succeeded, but the worker is still running.
            # WORKERS_RUNNING case.
            elif (
                pre_completion_status == py_trees.common.Status.SUCCESS
                and num_ticks <= pre_duration + worker_duration
            ):
                execution_case = ExecutionCase.WORKERS_RUNNING
                if num_ticks == pre_duration + 1:
                    # The pre terminates on the first tick after `pre_duration`
                    expected_counts[0] += 1
                    # The worker only gets re-initialized after the pre terminates.
                    expected_counts[1] = 0
                    # The pre's status gets set
                    expected_statuses[0] = pre_completion_status
                    expected_termination_new_statuses[0] = pre_completion_status
                expected_counts[1] += 1
                expected_statuses[1] = py_trees.common.Status.RUNNING
                root_expected_status = py_trees.common.Status.RUNNING
            # The pre succeeded and the worker has terminated.
            # POST_RUNNING case.
            elif (
                pre_completion_status == py_trees.common.Status.SUCCESS
                and num_ticks <= pre_duration + worker_duration + post_duration
            ):
                execution_case = ExecutionCase.POST_RUNNING
                if num_ticks == pre_duration + worker_duration + 1:
                    # The worker terminates on the first tick after `worker_duration`
                    expected_counts[1] += 1
                    # Post only gets reinitialized after the worker terminates.
                    expected_counts[2] = 0
                    # The worker status gets set
                    expected_statuses[1] = worker_completion_status
                    expected_termination_new_statuses[1] = worker_completion_status
                elif worker_completion_status == py_trees.common.Status.FAILURE:
                    # The Selector with memory unnecessarily sets previous children to
                    # INVALID the tick after they fail, hence the below switch.
                    # https://github.com/splintered-reality/py_trees/blob/0d5b39f2f6333c504406d8a63052c456c6bd1ce5/py_trees/composites.py#L427
                    for i in range(2):
                        expected_statuses[i] = py_trees.common.Status.INVALID
                        expected_termination_new_statuses[
                            i
                        ] = py_trees.common.Status.INVALID
                expected_counts[2] += 1
                expected_statuses[2] = py_trees.common.Status.RUNNING
                root_expected_status = py_trees.common.Status.RUNNING
            # The pre failed, but the post is still running.
            # POST_RUNNING case.
            elif (
                pre_completion_status == py_trees.common.Status.FAILURE
                and num_ticks <= pre_duration + post_duration
            ):
                execution_case = ExecutionCase.POST_RUNNING
                if num_ticks == pre_duration + 1:
                    # The pre terminates on the first tick after `pre_duration`
                    expected_counts[0] += 1
                    # Post only gets reinitialized after the worker terminates.
                    expected_counts[2] = 0
                    # The pre's status gets set
                    expected_statuses[0] = pre_completion_status
                    expected_termination_new_statuses[0] = pre_completion_status
                else:
                    # The Selector with memory unnecessarily sets previous children to
                    # INVALID the tick after they fail, hence the below switch.
                    # https://github.com/splintered-reality/py_trees/blob/0d5b39f2f6333c504406d8a63052c456c6bd1ce5/py_trees/composites.py#L427
                    expected_statuses[0] = py_trees.common.Status.INVALID
                    expected_termination_new_statuses[
                        0
                    ] = py_trees.common.Status.INVALID
                expected_counts[2] += 1
                expected_statuses[2] = py_trees.common.Status.RUNNING
                root_expected_status = py_trees.common.Status.RUNNING
            # The post has terminated. TREE_TERMINATED case.
            elif num_ticks == num_ticks_to_terminate:
                execution_case = ExecutionCase.TREE_TERMINATED
                expected_counts[2] += 1
                expected_statuses[2] = post_completion_status
                expected_termination_new_statuses[2] = post_completion_status
                root_expected_status = (
                    py_trees.common.Status.FAILURE
                    if pre_completion_status == py_trees.common.Status.FAILURE
                    else worker_completion_status
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
                termination_order = []
                for i in range(2):
                    if expected_statuses[i] != py_trees.common.Status.INVALID:
                        expected_statuses[i] = py_trees.common.Status.INVALID
                        expected_termination_new_statuses[
                            i
                        ] = py_trees.common.Status.INVALID
                        termination_order.append(behaviors[i])
                root_expected_status = py_trees.common.Status.INVALID
                # `post` should only get ticked on preemption if the worker/callback
                # have not yet terminated. If they have terminated, the root
                # is considered complete and there is no reason to run `post`.
                if execution_case != ExecutionCase.TREE_TERMINATED:
                    # `pose` should get ticked to completion
                    expected_counts[2] = post_duration + 1

                    # Because `post` is not officially a part of the tree,
                    # it won't get called as part of the preemption. So it's
                    # status will be its terminal status.
                    expected_statuses[2] = post_completion_status
                    expected_termination_new_statuses[2] = post_completion_status
                    termination_order.append(behaviors[2])

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
                check_termination_order(termination_order, descriptor)
                assert (
                    root.status == root_expected_status
                ), f"root status {root.status} is not {root_expected_status}, {descriptor}"

                # End this cycle
                break


################################################################################
# Test execution without preemptions
################################################################################


def test_pre_succeeds_worker_succeeds_post_succeeds():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.SUCCESS,
        worker_completion_status=py_trees.common.Status.SUCCESS,
        post_completion_status=py_trees.common.Status.SUCCESS,
    )


def test_pre_succeeds_worker_succeeds_post_fails():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.SUCCESS,
        worker_completion_status=py_trees.common.Status.SUCCESS,
        post_completion_status=py_trees.common.Status.FAILURE,
    )


def test_pre_succeeds_worker_fails_post_succeeds():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.SUCCESS,
        worker_completion_status=py_trees.common.Status.FAILURE,
        post_completion_status=py_trees.common.Status.SUCCESS,
    )


def test_pre_succeeds_worker_fails_post_fails():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.SUCCESS,
        worker_completion_status=py_trees.common.Status.FAILURE,
        post_completion_status=py_trees.common.Status.FAILURE,
    )


def test_pre_fails_worker_succeeds_post_succeeds():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.FAILURE,
        worker_completion_status=py_trees.common.Status.SUCCESS,
        post_completion_status=py_trees.common.Status.SUCCESS,
    )


def test_pre_fails_worker_succeeds_post_fails():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.FAILURE,
        worker_completion_status=py_trees.common.Status.SUCCESS,
        post_completion_status=py_trees.common.Status.FAILURE,
    )


def test_pre_fails_worker_fails_post_succeeds():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.FAILURE,
        worker_completion_status=py_trees.common.Status.FAILURE,
        post_completion_status=py_trees.common.Status.SUCCESS,
    )


def test_pre_fails_worker_fails_post_fails():
    """
    See `combined_test` docstring.
    """
    combined_test(
        pre_completion_status=py_trees.common.Status.FAILURE,
        worker_completion_status=py_trees.common.Status.FAILURE,
        post_completion_status=py_trees.common.Status.FAILURE,
    )
