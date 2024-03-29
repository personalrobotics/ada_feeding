"""
NOTE: This is a preempt-handling version of the idiom discussed in
https://github.com/splintered-reality/py_trees/pull/427 . Once a
preempt-handling version of that idiom is merged into py_trees, this
idiom should be removed in favor of the main py_trees one.
"""

import typing

from py_trees import behaviour, behaviours, composites

from ada_feeding.decorators import OnPreempt


def eventually_swiss(
    name: str,
    workers: typing.List[behaviour.Behaviour],
    on_failure: behaviour.Behaviour,
    on_success: behaviour.Behaviour,
    on_preempt: behaviour.Behaviour,
    on_preempt_single_tick: bool = True,
    on_preempt_period_ms: int = 0,
    on_preempt_timeout: typing.Optional[float] = None,
    return_on_success_status: bool = True,
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
       root behaviour returned from this idiom while the root behaviour's status is
       :data:`~py_trees.common.Status.RUNNING`.

    The return status of this idiom in non-preemption cases is:
    - If the workers all return SUCCESS:
        - If `return_on_success_status` is True, then the status of the root behaviour
          returned from this idiom is status of `on_success`.
        - If `return_on_success_status` is False, then the status of the root behaviour
          returned from this idiom is :data:`~py_trees.common.Status.SUCCESS`.
    - If at least one worker returns FAILURE, return :data:`~py_trees.common.Status.FAILURE`.

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
        return_on_success_status: if True, pass the `on_success` status to the
            root, else return :data:`~py_trees.common.Status.SUCCESS`.

    Returns:
        :class:`~py_trees.behaviour.Behaviour`: the root of the eventually_swiss subtree

    .. seealso:: :meth:`py_trees.idioms.eventually`, :ref:`py-trees-demo-eventually-swiss-program`
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # This is acceptable, to give users maximum control over how this swiss-knife
    # idiom behaves.
    # pylint: disable=abstract-class-instantiated
    # behaviours.Failure and behaviours.Success are valid instantiations

    workers_sequence = composites.Sequence(
        name="Workers",
        memory=True,
        children=workers,
    )
    on_failure_return_status = composites.Sequence(
        name="On Failure Return Failure",
        memory=True,
        children=[on_failure, behaviours.Failure(name="Failure")],
    )
    on_failure_subtree = composites.Selector(
        name="On Failure",
        memory=True,
        children=[workers_sequence, on_failure_return_status],
    )
    if return_on_success_status:
        on_success_return_status = on_success
    else:
        on_success_return_status = composites.Selector(
            name="On Success Return Success",
            memory=True,
            children=[on_success, behaviours.Success(name="Success")],
        )
    on_success_subtree = composites.Sequence(
        name="On Success",
        memory=True,
        children=[on_failure_subtree, on_success_return_status],
    )
    root = OnPreempt(
        name=name,
        child=on_success_subtree,
        on_preempt=on_preempt,
        single_tick=on_preempt_single_tick,
        period_ms=on_preempt_period_ms,
        timeout=on_preempt_timeout,
    )

    return root
