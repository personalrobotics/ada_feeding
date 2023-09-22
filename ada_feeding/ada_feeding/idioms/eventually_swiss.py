"""
NOTE: This is a preempt-handling version of the idiom discussed in
https://github.com/splintered-reality/py_trees/pull/427 . Once a
preempt-handling version of that idiom is merged into py_trees, this
idiom should be removed in favor of the main py_trees one.
"""

import typing

from py_trees import behaviour, behaviours, common, composites
from py_trees.decorators import Inverter, StatusToBlackboard, SuccessIsFailure

from ada_feeding.decorators import ForceStatus, OnPreempt


def eventually_swiss(
    name: str,
    workers: typing.List[behaviour.Behaviour],
    on_failure: behaviour.Behaviour,
    on_success: behaviour.Behaviour,
    on_preempt: behaviour.Behaviour,
    on_preempt_single_tick: bool = True,
    on_preempt_period_ms: int = 0,
    on_preempt_timeout: typing.Optional[float] = None,
    status_blackboard_key: typing.Optional[str] = None,
) -> behaviour.Behaviour:
    """
    Implement a multi-tick, general purpose 'try-except-else'-like pattern.

    This is a swiss knife version of the eventually idiom
    that facilitates a multi-tick response for specialised
    handling work sequence's completion status. Specifically, this idiom
    guarentees the following:
    1. The on_success behaviour is ticked only if the workers all return SUCCESS.
    2. The on_failure behaviour is ticked only if at least one worker returns FAILURE.
    3. The on_preempt behaviour is ticked only if `stop(INVALID)` is called on the
       root behavior returned from this idiom.

    .. graphviz:: dot/eventually-swiss.dot

    Args:
        name: the name to use for the idiom root
        workers: the worker behaviours or subtrees
        on_success: the behaviour or subtree to tick on work success
        on_failure: the behaviour or subtree to tick on work failure
        on_preempt: the behaviour or subtree to tick on work preemption
        on_preempt_single_tick: if True, tick the on_preempt behaviour once
            on preemption. Else, tick the on_preempt behaviour until it
            reaches a status other than :data:`~py_trees.common.Status.RUNNING`.
        on_preempt_period_ms: how long to sleep between ticks (in milliseconds)
            if `on_preempt_single_tick` is False. If 0, then do not sleep.
        on_preempt_timeout: how long (sec) to wait for the on_preempt behaviour
            to reach a status other than :data:`~py_trees.common.Status.RUNNING`
            if `on_preempt_single_tick` is False. If None, then do not timeout.
        status_blackboard_key: the key to use for the status blackboard variable.
            If None, use "/{name}/eventually_swiss_status".

    Returns:
        :class:`~py_trees.behaviour.Behaviour`: the root of the oneshot subtree

    .. seealso:: :meth:`py_trees.idioms.eventually`, :ref:`py-trees-demo-eventually-swiss-program`
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # This is acceptable, to give users maximum control over how the swiss knife
    # idiom behaves.

    # Create the subtree to handle `on_success`
    if status_blackboard_key is None:
        status_blackboard_key = f"/{name}/eventually_swiss_status"
    unset_status = behaviours.UnsetBlackboardVariable(
        name="Unset Status", key=status_blackboard_key
    )
    save_success_status = StatusToBlackboard(
        name="Save Success Status",
        child=on_success,
        variable_name=status_blackboard_key,
    )
    on_success_sequence = composites.Sequence(
        name="Work to Success",
        memory=True,
        children=[unset_status] + workers + [save_success_status],
    )

    # Create the subtree to handle `on_failure`. This subtree must start with
    # `return_status_if_set` so that `on_failure` is not ticked if `on_success`
    # fails.
    check_status_exists = behaviours.CheckBlackboardVariableExists(
        name="Wait for Status",
        variable_name=status_blackboard_key,
    )
    # Note that we can get away with using an Inverter here because the only way
    # we get to this branch is if either the `workers` or `on_success` fails.
    # So the status either doesn't exist or is FAILURE.
    return_status_if_set = Inverter(
        name="Return Status if Set",
        child=check_status_exists,
    )
    on_failure_always_fail = SuccessIsFailure(
        name="On Failure Always Failure",
        child=on_failure,
    )
    save_failure_status = StatusToBlackboard(
        name="Save Failure Status",
        child=on_failure_always_fail,
        variable_name=status_blackboard_key,
    )
    on_failure_sequence = composites.Sequence(
        name="On Failure",
        memory=True,
        children=[return_status_if_set, save_failure_status],
    )

    # Create the combined subtree to handle `on_success` and `on_failure`
    combined = composites.Selector(
        name="On Non-Preemption Subtree",
        memory=True,
        children=[on_success_sequence, on_failure_sequence],
    )
    # We force the outcome of this tree to FAILURE so that the Selector always
    # goes on to tick the `on_preempt_subtree`, which is necessary to ensure
    # that the `on_preempt` behavior will get run if the tree is preempted.
    on_success_or_failure_subtree = ForceStatus(
        name="On Non-Preemption",
        child=combined,
        status=common.Status.FAILURE,
    )

    # Create the subtree to handle `on_preempt`
    on_preempt_subtree = OnPreempt(
        name="On Preemption",
        child=on_preempt,
        # Returning FAILURE is necessary so (a) the Selector always moves on to
        # `set_status_subtree`; and (b) the decorator's status is not INVALID (which would
        # prevent the Selector from passing on a `stop(INVALID)` call to it).
        update_status=common.Status.FAILURE,
        single_tick=on_preempt_single_tick,
        period_ms=on_preempt_period_ms,
        timeout=on_preempt_timeout,
    )

    # Create the subtree to output the status once `on_success_or_failure_subtree`
    # is done.
    wait_for_status = behaviours.WaitForBlackboardVariable(
        name="Wait for Status",
        variable_name=status_blackboard_key,
    )
    blackboard_to_status = behaviours.BlackboardToStatus(
        name="Blackboard to Status",
        variable_name=status_blackboard_key,
    )
    set_status_subtree = composites.Sequence(
        name="Set Status",
        memory=True,
        children=[wait_for_status, blackboard_to_status],
    )

    root = composites.Selector(
        name=name,
        memory=False,
        children=[
            on_success_or_failure_subtree,
            on_preempt_subtree,
            set_status_subtree,
        ],
    )
    return root
