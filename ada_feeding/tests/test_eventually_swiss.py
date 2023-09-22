#!/usr/bin/env python3
"""
This module defines unit tests for the eventually_swiss idiom.
"""

# Standard imports
from enum import Enum
import time
from typing import List, Optional

# Third-party imports
import py_trees

# Local imports
from ada_feeding.idioms import eventually_swiss


class TickCounterWithTerminateTimestamp(py_trees.behaviours.TickCounter):
    """
    This class is identical to TickCounter, except that it also stores the
    timestamp when the behavior terminated.
    """

    def __init__(
        self, name: str, duration: int, completion_status: py_trees.common.Status
    ):
        """
        Initialise the behavior.
        """
        super().__init__(
            name=name, duration=duration, completion_status=completion_status
        )
        self.termination_new_status = None
        self.termination_timestamp = None

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Terminate the behavior.
        """
        self.termination_new_status = new_status
        self.termination_timestamp = time.time()


def check_count_status(
    behaviors: List[py_trees.behaviours.TickCounter],
    counts: List[int],
    statuses: List[py_trees.common.Status],
    descriptor: str = "",
) -> None:
    """
    Takes in a list of TickCounter behaviors and checks that their counts and
    statuses are correct.

    Parameters
    ----------
    behaviors: The list of behaviors to check.
    counts: The expected counts for each behavior.
    statuses: The expected statuses for each behavior.
    """
    assert (
        len(behaviors) == len(counts) == len(statuses)
    ), "lengths of behaviors, counts, and statuses must be equal"
    for i, behavior in enumerate(behaviors):
        assert behavior.counter == counts[i], (
            f"behavior '{behavior.name}' actual count {behavior.counter}, "
            f"expected count {counts[i]}, "
            f"{descriptor}"
        )
        assert behavior.status == statuses[i], (
            f"behavior '{behavior.name}' actual status {behavior.status}, "
            f"expected status {statuses[i]}, "
            f"{descriptor}"
        )


def check_termination_new_statuses(
    behaviors: List[TickCounterWithTerminateTimestamp],
    statuses: List[Optional[py_trees.common.Status]],
    descriptor: str = "",
) -> None:
    """
    Checkes that `terminate` either has not been called on the behavior, or
    that it has been called with the correct new status.

    Parameters
    ----------
    behaviors: The list of behaviors to check.
    statuses: The expected new statuses for each behavior when `terminate` was
        called, or `None` if `terminate` was not expected to be called.
    """
    assert len(behaviors) == len(
        statuses
    ), "lengths of behaviors and statuses must be equal"
    for i, behavior in enumerate(behaviors):
        if statuses[i] is None:
            assert behavior.termination_new_status is None, (
                f"behavior '{behavior.name}' expected termination_new_status None, actual "
                f"termination_new_status {behavior.termination_new_status}, "
                f"{descriptor}"
            )
        else:
            assert behavior.termination_new_status == statuses[i], (
                f"behavior '{behavior.name}' actual termination_new_status "
                f"{behavior.termination_new_status}, expected termination_new_status "
                f"{statuses[i]}, {descriptor}"
            )


def check_termination_order(
    behaviors: List[TickCounterWithTerminateTimestamp],
    descriptor: str = "",
) -> None:
    """
    Checks that the behaviors terminated in the correct order.

    Parameters
    ----------
    behaviors: The list of behaviors to check, in the order that `terminate`
        should have been called on them.
    """
    for i in range(len(behaviors) - 1):
        assert (
            behaviors[i].termination_timestamp <= behaviors[i + 1].termination_timestamp
        ), (
            f"behavior '{behaviors[i].name}' terminated after behavior "
            f"'{behaviors[i + 1].name}', when it should have terminated before, "
            f"{descriptor}"
        )


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

    Note that this always generates the multi-tick version of eventually_swiss.
    """
    # pylint: disable=too-many-arguments
    # Necessary to create a versatile test generation function.

    # Setup the test
    worker = TickCounterWithTerminateTimestamp(
        name="Worker",
        duration=worker_duration,
        completion_status=worker_completion_status,
    )
    on_success = TickCounterWithTerminateTimestamp(
        name="On Success",
        duration=on_success_duration,
        completion_status=on_success_completion_status,
    )
    on_failure = TickCounterWithTerminateTimestamp(
        name="On Failure",
        duration=on_failure_duration,
        completion_status=on_failure_completion_status,
    )
    on_preempt = TickCounterWithTerminateTimestamp(
        name="On Preempt",
        duration=on_preempt_duration,
        completion_status=on_preempt_completion_status,
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


class ExecutionCase(Enum):
    """
    Tree execution can broadly fall into one of the below three cases.
    """

    WORKER_RUNNING = 0
    WORKER_TERMINATED_CALLBACK_RUNNING = 1
    WORKER_TERMINATED_CALLBACK_TERMINATED = 2


def combined_test(
    worker_completion_status: py_trees.common.Status,
    callback_completion_status: py_trees.common.Status,
    global_num_cycles: int = 2,
    preempt_times: List[Optional[ExecutionCase]] = [None, None],
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
        Case WORKER_TERMINATED_CALLBACK_TERMINATED:
         - When `on_success` returns `callback_completion_status`, `on_failure`
           and `on_preempt` should not be ticked, none of the functions should
           be terminated, and the root should return `callback_completion_status`.

    When the worker fails:
        Case WORKER_TERMINATED_CALLBACK_RUNNING:
         - While `on_failure` is RUNNING, `on_success` and `on_preempt` should
           not be ticked, none of the functions should be terminated, and the
           root should be running.
        Case WORKER_TERMINATED_CALLBACK_TERMINATED:
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
      - WORKER_TERMINATED_CALLBACK_TERMINATED: Terminate the tree after the tick
        when the worker has terminated and the callback has terminated (i.e., after
        the tick where the root returns a non-RUNNING status).
    After terminating the tree, this function checks that the tree has not ticked
    `worker`, `on_success`, or `on_failure` any more, but has ticked `on_preempt`
    to completion. It also checks that the tree has terminated in the correct order:
    `worker` -> `on_success`/`on_failure` -> `on_preempt`.

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
        # The worker and on_success get reset to INVALID at the beginning of
        # every cycle, since they are in the first sequence
        if num_cycles > 0:
            expected_statuses[0] = py_trees.common.Status.INVALID
            expected_termination_new_statuses[0] = py_trees.common.Status.INVALID
            if (
                worker_completion_status == py_trees.common.Status.SUCCESS
                and expected_statuses[1] != py_trees.common.Status.INVALID
            ):
                expected_termination_new_statuses[1] = py_trees.common.Status.INVALID
                expected_statuses[1] = py_trees.common.Status.INVALID
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
                    # on_failure is only set to INVALID after the worker terminates
                    if (
                        num_cycles > 0
                        and worker_completion_status == py_trees.common.Status.FAILURE
                        and expected_statuses[2] != py_trees.common.Status.INVALID
                    ):
                        expected_termination_new_statuses[
                            2
                        ] = py_trees.common.Status.INVALID
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
            # WORKER_TERMINATED_CALLBACK_TERMINATED case.
            elif num_ticks == worker_duration + callback_duration + 1:
                execution_case = ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED
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

                # Update the expected counts, statuses, termination_new_statuses, and
                # root status following the preemption.
                expected_counts[3] = callback_duration + 1
                # Update the expected termination of all non-invalid behaviors.
                # We have separate termination orders for on_success and on_failure
                # because we don't care about the relative order of the two, all
                # we care is that worker terminates before them and `on_preempt`
                # is run and terminates after them.
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
                # Because `on_preempt` also got preempted, its status is INVALID
                # even though it ran to completion (as indicated by `expected_countsthe count`)
                expected_statuses[3] = py_trees.common.Status.INVALID
                expected_termination_new_statuses[3] = py_trees.common.Status.INVALID
                root_expected_status = py_trees.common.Status.INVALID
                termination_order_on_success.append(behaviors[3])
                termination_order_on_failure.append(behaviors[3])
                descriptor += " after preemption"

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
# Test execution without preemptions
################################################################################


def test_worker_succeeds_on_success_succeeds():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
    )


def test_worker_succeeds_on_success_fails():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.FAILURE,
    )


def test_worker_fails_on_failure_succeeds():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.SUCCESS,
    )


def test_worker_fails_on_failure_fails():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
    )


################################################################################
# Test execution with one preemption, followed by a full run-through
################################################################################


def test_first_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[ExecutionCase.WORKER_RUNNING, None],
    )


def test_first_preempt_while_callback_running_success():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING, None],
    )


def test_first_preempt_while_callback_running_fail():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING, None],
    )


def test_first_preempt_while_callback_terminated_success_success():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED, None],
    )


def test_first_preempt_while_callback_terminated_success_fail():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED, None],
    )


def test_first_preempt_while_callback_terminated_fail_success():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED, None],
    )


def test_first_preempt_while_callback_terminated_fail_fail():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED, None],
    )


################################################################################
# Test execution with a full run-through, followed by a preemption
################################################################################


def test_second_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[None, ExecutionCase.WORKER_RUNNING],
    )


def test_second_preempt_while_callback_running_success():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[None, ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING],
    )


def test_second_preempt_while_callback_running_fail():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[None, ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING],
    )


def test_second_preempt_while_callback_terminated_success_success():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[None, ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED],
    )


def test_second_preempt_while_callback_terminated_success_fail():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[None, ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED],
    )


def test_second_preempt_while_callback_terminated_fail_success():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[None, ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED],
    )


def test_second_preempt_while_callback_terminated_fail_fail():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[None, ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED],
    )


################################################################################
# Test execution with two consecutive preemptions
################################################################################


def test_success_preempt_while_worker_running_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[ExecutionCase.WORKER_RUNNING, ExecutionCase.WORKER_RUNNING],
    )


def test_success_preempt_while_worker_running_preempt_while_callback_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
        ],
    )


def test_success_preempt_while_worker_running_preempt_while_callback_terminated():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
        ],
    )


def test_success_preempt_while_callback_running_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
            ExecutionCase.WORKER_RUNNING,
        ],
    )


def test_success_preempt_while_callback_running_preempt_while_callback_running_success():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
        ],
    )


def test_success_preempt_while_callback_running_preempt_while_callback_terminated():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
        ],
    )


def test_success_preempt_while_callback_terminated_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
            ExecutionCase.WORKER_RUNNING,
        ],
    )


def test_success_preempt_while_callback_terminated_preempt_while_callback_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
        ],
    )


def test_success_preempt_while_callback_terminated_preempt_while_callback_terminated():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.SUCCESS,
        callback_completion_status=py_trees.common.Status.SUCCESS,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
        ],
    )


def test_fail_preempt_while_worker_running_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[ExecutionCase.WORKER_RUNNING, ExecutionCase.WORKER_RUNNING],
    )


def test_fail_preempt_while_worker_running_preempt_while_callback_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
        ],
    )


def test_fail_preempt_while_worker_running_preempt_while_callback_terminated():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
        ],
    )


def test_fail_preempt_while_callback_running_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
            ExecutionCase.WORKER_RUNNING,
        ],
    )


def test_fail_preempt_while_callback_running_preempt_while_callback_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
        ],
    )


def test_fail_preempt_while_callback_running_preempt_while_callback_terminated():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
        ],
    )


def test_fail_preempt_while_callback_terminated_preempt_while_worker_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
            ExecutionCase.WORKER_RUNNING,
        ],
    )


def test_fail_preempt_while_callback_terminated_preempt_while_callback_running():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_RUNNING,
        ],
    )


def test_fail_preempt_while_callback_terminated_preempt_while_callback_terminated():
    """
    See docsting for `combined_test`.
    """
    combined_test(
        worker_completion_status=py_trees.common.Status.FAILURE,
        callback_completion_status=py_trees.common.Status.FAILURE,
        preempt_times=[
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
            ExecutionCase.WORKER_TERMINATED_CALLBACK_TERMINATED,
        ],
    )
