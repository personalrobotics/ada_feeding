#!/usr/bin/env python3
"""
This module defines unit tests for the scoped_behaviour idiom.
"""

# Standard imports
from enum import Enum
from functools import partial
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

# pylint: disable=duplicate-code
# `test_scoped_behavior` and `test_eventually_swiss` have similar code because
# they are similar idioms. That is okay.
# pylint: disable=redefined-outer-name
# When generating tests, we use global variables with the same names as
# variables in the functions. That is okay, since the functions don't need
# access to the global variables.


class ExecutionCase(Enum):
    """
    Tree execution can broadly fall into one of the below cases.
    """

    NONE = 0
    HASNT_STARTED = 1
    PRE_RUNNING = 2
    WORKERS_RUNNING = 3
    POST_RUNNING = 4
    TREE_TERMINATED = 5


def generate_test(
    pre_duration: int,
    pre_completion_status: py_trees.common.Status,
    worker_duration: int,
    worker_completion_status: py_trees.common.Status,
    post_duration: int,
    post_completion_status: py_trees.common.Status,
    worker_override: Optional[py_trees.behaviour.Behaviour] = None,
    suffix: str = ",",
):
    """
    Generates a worker, pre, and post behavior with the
    specified durations and completion statuses.

    Parameters
    ----------
    pre_duration: The duration of the pre behavior.
    pre_completion_status: The completion status of the pre behavior.
    worker_duration: The duration of the worker behavior.
    worker_completion_status: The completion status of the worker behavior.
    post_duration: The duration of the post behavior.
    post_completion_status: The completion status of the post behavior.
    worker_override: If not None, this behavior will be used instead of the
        default worker behavior.
    """
    # pylint: disable=too-many-arguments
    # Necessary to create a versatile test generation function.

    # Setup the test
    pre = TickCounterWithTerminateTimestamp(
        name="Pre" + suffix,
        duration=pre_duration,
        completion_status=pre_completion_status,
        ns="/pre" + suffix,
    )
    if worker_override is None:
        worker = TickCounterWithTerminateTimestamp(
            name="Worker" + suffix,
            duration=worker_duration,
            completion_status=worker_completion_status,
            ns="/worker" + suffix,
        )
    else:
        worker = worker_override

    post = TickCounterWithTerminateTimestamp(
        name="Post" + suffix,
        duration=post_duration,
        completion_status=post_completion_status,
        ns="/post" + suffix,
    )

    root = scoped_behavior(
        name="Root" + suffix,
        pre_behavior=pre,
        workers=[worker],
        post_behavior=post,
    )
    return root, pre, worker, post


def combined_test(
    pre_completion_status: py_trees.common.Status,
    worker_completion_status: py_trees.common.Status,
    post_completion_status: py_trees.common.Status,
    global_num_cycles: int = 2,
    preempt_times: List[ExecutionCase] = [ExecutionCase.NONE, ExecutionCase.NONE],
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
    root, pre, worker, post = generate_test(
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
    behaviors = [pre, worker, post]
    expected_counts = [0, 0, 0]
    expected_statuses = [
        py_trees.common.Status.INVALID,
        py_trees.common.Status.INVALID,
        py_trees.common.Status.INVALID,
    ]
    expected_num_times_ticked_to_non_running_statuses = [0, 0, 0]
    expected_termination_new_statuses = [None, None, None]

    # Tick the tree
    preempted_in_previous_cycle = False
    for num_cycles in range(global_num_cycles):
        execution_case = ExecutionCase.HASNT_STARTED
        for num_ticks in range(1, num_ticks_to_terminate + 2):
            descriptor = f"num_ticks {num_ticks}, num_cycles {num_cycles}"

            # Preempt if requested
            if preempt_times[num_cycles] == execution_case:
                root.stop(py_trees.common.Status.INVALID)
                descriptor += " after preemption"

                # Update the expected termination of all behaviors but `post`
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
                # is considered complete and there is no reason to run `post` again.
                if execution_case == ExecutionCase.HASNT_STARTED:
                    # In this cases, `post` should not get ticked as part of
                    # preemption. Its status will only be set to INVALID if neither
                    # it nor its parent is INVALID. The only case where `post` is
                    # not INVALID but its parent is is if the tree was preempted
                    # in the previous cycle.
                    if (
                        expected_statuses[2] != py_trees.common.Status.INVALID
                        and not preempted_in_previous_cycle
                    ):
                        expected_statuses[2] = py_trees.common.Status.INVALID
                        expected_termination_new_statuses[
                            2
                        ] = py_trees.common.Status.INVALID
                    preempted_in_previous_cycle = False
                elif execution_case == ExecutionCase.TREE_TERMINATED:
                    # In this cases, `post` should not get ticked as part of
                    # preemption. Its status will be set to INVALID through the
                    # normal termination process.
                    expected_statuses[2] = py_trees.common.Status.INVALID
                    expected_termination_new_statuses[
                        2
                    ] = py_trees.common.Status.INVALID
                else:
                    preempted_in_previous_cycle = True
                    # `post` should get ticked to completion
                    expected_counts[2] = post_duration + 1

                    # Because `post` is not officially a part of the tree,
                    # it won't get called as part of the preemption. So it's
                    # status will be its terminal status.
                    expected_statuses[2] = post_completion_status
                    expected_num_times_ticked_to_non_running_statuses[2] += 1
                    expected_termination_new_statuses[2] = post_completion_status
                termination_order.append(behaviors[2])

                # Run the preemption tests
                check_count_status(
                    behaviors=behaviors,
                    counts=expected_counts,
                    statuses=expected_statuses,
                    num_times_ticked_to_non_running_statuses=(
                        expected_num_times_ticked_to_non_running_statuses
                    ),
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

            if num_ticks == num_ticks_to_terminate + 1:
                # End this cycle. We only go past the ticks to terminate in case
                # the tree is preempted after termination.
                preempted_in_previous_cycle = False
                break
            if num_ticks == 1:
                # The pre's count gets re-initialized at the beginning of every cycle.
                expected_counts[0] = 0
                # The pre and worker get reset to INVALID at the
                # beginning of every cycle. Post gets reset to INVALID only if the
                # last cycle was not preempted.
                if num_cycles > 0:
                    for i in range(3):
                        if i < 2 or preempt_times[num_cycles - 1] == ExecutionCase.NONE:
                            expected_statuses[i] = py_trees.common.Status.INVALID
                            if expected_termination_new_statuses[i] is not None:
                                expected_termination_new_statuses[
                                    i
                                ] = py_trees.common.Status.INVALID

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
                    expected_num_times_ticked_to_non_running_statuses[0] += 1
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
                    expected_num_times_ticked_to_non_running_statuses[1] += 1
                    expected_termination_new_statuses[1] = worker_completion_status
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
                    expected_num_times_ticked_to_non_running_statuses[0] += 1
                    expected_termination_new_statuses[0] = pre_completion_status
                expected_counts[2] += 1
                expected_statuses[2] = py_trees.common.Status.RUNNING
                root_expected_status = py_trees.common.Status.RUNNING
            # The post has terminated. TREE_TERMINATED case.
            elif num_ticks == num_ticks_to_terminate:
                execution_case = ExecutionCase.TREE_TERMINATED
                expected_counts[2] += 1
                expected_statuses[2] = post_completion_status
                expected_num_times_ticked_to_non_running_statuses[2] += 1
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
                num_times_ticked_to_non_running_statuses=(
                    expected_num_times_ticked_to_non_running_statuses
                ),
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


################################################################################
# Generate all tests with 2 cycles
################################################################################

# Set the status cases to iterate over
status_cases = [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]
for pre_completion_status in status_cases:
    # Set the preempt cases to iterate over
    preempt_cases = [
        ExecutionCase.NONE,
        ExecutionCase.HASNT_STARTED,
        ExecutionCase.PRE_RUNNING,
    ]
    if pre_completion_status == py_trees.common.Status.SUCCESS:
        preempt_cases.append(ExecutionCase.WORKERS_RUNNING)
    preempt_cases += [ExecutionCase.POST_RUNNING, ExecutionCase.TREE_TERMINATED]

    for worker_completion_status in status_cases:
        for post_completion_status in status_cases:
            for first_preempt in preempt_cases:
                for second_preempt in preempt_cases:
                    test_name = (
                        f"test_pre_{pre_completion_status.name}_worker_"
                        f"{worker_completion_status.name}_post_{post_completion_status.name}_"
                        f"first_preempt_{first_preempt.name}_second_preempt_{second_preempt.name}"
                    )
                    globals()[test_name] = partial(
                        combined_test,
                        pre_completion_status=pre_completion_status,
                        worker_completion_status=worker_completion_status,
                        post_completion_status=post_completion_status,
                        preempt_times=[first_preempt, second_preempt],
                    )

################################################################################
# Test Nested Scoped Behaviors
################################################################################


class NestedExecutionCase(Enum):
    """
    With a single nested sequence, execution can broadly fall into one of the
    below cases.
    """

    NONE = 0
    HASNT_STARTED = 1
    PRE1_RUNNING = 2
    PRE2_RUNNING = 3
    WORKERS_RUNNING = 4
    POST1_RUNNING = 5
    POST2_RUNNING = 6
    TREE_TERMINATED = 7


def nested_behavior_tests(
    preempt_time: NestedExecutionCase,
):
    """
    In the test of nested scope, we will assume all behaviors succeed, because
    success/failure was already tested above. We will also only tick the tree
    for one cycle, because multiple cycles were tested above. The main goal of
    this test is to ensure the following:
    - NONE: If the tree is not preempted, both post-behaviors should be ticked
      to completion.
    - PRE1_RUNNING: If the tree is preempted while pre1 is running, post1 should
      be ticked to completion, and post2 should not be ticked.
    - PRE2_RUNNING, WORKERS_RUNNING, POST1_RUNNING, POST2_RUNNING: In all of
      these cases, post1 and post2 should be ticked to completion.
    - TREE_TERMINATED: If the tree is preempted after the tree has terminated,
      post1 and post2 should not be ticked.
    """
    # pylint: disable=too-many-branches, too-many-statements
    # Necessary to test all the cases

    pre1_duration = 3
    pre2_duration = 2
    worker_duration = 2
    post1_duration = 6
    post2_duration = 4
    worker_override, pre2, worker, post2 = generate_test(
        pre_duration=pre2_duration,
        pre_completion_status=py_trees.common.Status.SUCCESS,
        worker_duration=worker_duration,
        worker_completion_status=py_trees.common.Status.SUCCESS,
        post_duration=post2_duration,
        post_completion_status=py_trees.common.Status.SUCCESS,
        suffix="2",
    )
    root, pre1, _, post1 = generate_test(
        pre_duration=pre1_duration,
        pre_completion_status=py_trees.common.Status.SUCCESS,
        worker_duration=0,
        worker_completion_status=py_trees.common.Status.INVALID,
        post_duration=post1_duration,
        post_completion_status=py_trees.common.Status.SUCCESS,
        worker_override=worker_override,
        suffix="1",
    )
    behaviors = [pre1, pre2, worker, post2, post1]

    # Get the number of ticks to terminate the tree
    num_ticks_to_terminate = (
        pre1_duration
        + pre2_duration
        + worker_duration
        + post1_duration
        + post2_duration
        + 1
    )

    if preempt_time == NestedExecutionCase.NONE:
        for _ in range(num_ticks_to_terminate):
            root.tick_once()
        check_count_status(
            behaviors=behaviors,
            counts=[
                pre1_duration + 1,
                pre2_duration + 1,
                worker_duration + 1,
                post2_duration + 1,
                post1_duration + 1,
            ],
            statuses=[py_trees.common.Status.SUCCESS] * 5,
            num_times_ticked_to_non_running_statuses=[1] * 5,
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[py_trees.common.Status.SUCCESS] * 5,
        )
        check_termination_order(behaviors)
        assert (
            root.status == py_trees.common.Status.SUCCESS
        ), f"root status {root.status} is not SUCCESS"
    elif preempt_time == NestedExecutionCase.HASNT_STARTED:
        root.stop(py_trees.common.Status.INVALID)
        check_count_status(
            behaviors=behaviors,
            counts=[0] * 5,
            statuses=[py_trees.common.Status.INVALID] * 5,
            num_times_ticked_to_non_running_statuses=[0] * 5,
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[None] * 5,
        )
    elif preempt_time == NestedExecutionCase.PRE1_RUNNING:
        for _ in range(1):
            root.tick_once()
        root.stop(py_trees.common.Status.INVALID)
        check_count_status(
            behaviors=behaviors,
            counts=[1, 0, 0, 0, post1_duration + 1],
            statuses=[py_trees.common.Status.INVALID] * 4
            + [py_trees.common.Status.SUCCESS],
            num_times_ticked_to_non_running_statuses=[0] * 4 + [1],
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[py_trees.common.Status.INVALID]
            + [None] * 3
            + [py_trees.common.Status.SUCCESS],
        )
        check_termination_order([pre1, post1])
    elif preempt_time == NestedExecutionCase.PRE2_RUNNING:
        for _ in range(pre1_duration + 1):
            root.tick_once()
        root.stop(py_trees.common.Status.INVALID)
        check_count_status(
            behaviors=behaviors,
            counts=[pre1_duration + 1, 1, 0, post2_duration + 1, post1_duration + 1],
            statuses=[py_trees.common.Status.INVALID] * 3
            + [py_trees.common.Status.SUCCESS] * 2,
            num_times_ticked_to_non_running_statuses=[1] + [0] * 2 + [1] * 2,
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[py_trees.common.Status.INVALID] * 2
            + [None]
            + [py_trees.common.Status.SUCCESS] * 2,
        )
        check_termination_order([pre2, post2, post1])
    elif preempt_time == NestedExecutionCase.WORKERS_RUNNING:
        for _ in range(pre1_duration + pre2_duration + 1):
            root.tick_once()
        root.stop(py_trees.common.Status.INVALID)
        check_count_status(
            behaviors=behaviors,
            counts=[
                pre1_duration + 1,
                pre2_duration + 1,
                1,
                post2_duration + 1,
                post1_duration + 1,
            ],
            statuses=[py_trees.common.Status.INVALID] * 3
            + [py_trees.common.Status.SUCCESS] * 2,
            num_times_ticked_to_non_running_statuses=[1] * 2 + [0] + [1] * 2,
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[py_trees.common.Status.INVALID] * 3
            + [py_trees.common.Status.SUCCESS] * 2,
        )
        check_termination_order([pre1, pre2, worker, post2, post1])
    elif preempt_time == NestedExecutionCase.POST2_RUNNING:
        for _ in range(pre1_duration + pre2_duration + worker_duration + 1):
            root.tick_once()
        root.stop(py_trees.common.Status.INVALID)
        check_count_status(
            behaviors=behaviors,
            counts=[
                pre1_duration + 1,
                pre2_duration + 1,
                worker_duration + 1,
                post2_duration + 1,
                post1_duration + 1,
            ],
            statuses=[py_trees.common.Status.INVALID] * 3
            + [py_trees.common.Status.SUCCESS] * 2,
            num_times_ticked_to_non_running_statuses=[1] * 3 + [1] * 2,
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[py_trees.common.Status.INVALID] * 3
            + [py_trees.common.Status.SUCCESS] * 2,
        )
        check_termination_order([pre1, pre2, worker, post2, post1])
    elif preempt_time == NestedExecutionCase.POST1_RUNNING:
        for _ in range(
            pre1_duration + pre2_duration + worker_duration + post2_duration + 1
        ):
            root.tick_once()
        root.stop(py_trees.common.Status.INVALID)
        check_count_status(
            behaviors=behaviors,
            counts=[
                pre1_duration + 1,
                pre2_duration + 1,
                worker_duration + 1,
                post2_duration + 1,
                post1_duration + 1,
            ],
            # This is crucial -- POST2 should get terminated through the standard means,
            # not through OnPreempt.
            statuses=[py_trees.common.Status.INVALID] * 4
            + [py_trees.common.Status.SUCCESS],
            # This is crucial -- POST2 should only get ticked to completion once.
            num_times_ticked_to_non_running_statuses=[1] * 5,
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[py_trees.common.Status.INVALID] * 4
            + [py_trees.common.Status.SUCCESS],
        )
        check_termination_order([pre1, pre2, worker, post2, post1])
    elif preempt_time == NestedExecutionCase.TREE_TERMINATED:
        for _ in range(num_ticks_to_terminate):
            root.tick_once()
        root.stop(py_trees.common.Status.INVALID)
        check_count_status(
            behaviors=behaviors,
            counts=[
                pre1_duration + 1,
                pre2_duration + 1,
                worker_duration + 1,
                post2_duration + 1,
                post1_duration + 1,
            ],
            statuses=[py_trees.common.Status.INVALID] * 5,
            num_times_ticked_to_non_running_statuses=[1] * 5,
        )
        check_termination_new_statuses(
            behaviors=behaviors,
            statuses=[py_trees.common.Status.INVALID] * 5,
        )
        check_termination_order([pre1, pre2, worker, post2, post1])


for preempt_time in NestedExecutionCase:
    test_name = f"test_nested_{preempt_time.name}"
    globals()[test_name] = partial(
        nested_behavior_tests,
        preempt_time=preempt_time,
    )
