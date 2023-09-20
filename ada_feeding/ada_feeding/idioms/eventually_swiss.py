"""
NOTE: This is a preempt-handling version of the idiom discussed in
https://github.com/splintered-reality/py_trees/pull/427 . Once a
preempt-handling version of that idiom is merged into py_trees, this
idiom should be removed in favor of the main py_trees one.
"""

import operator
import typing

from py_trees import behaviour, behaviours, blackboard, common, composites, decorators

def eventually_swiss(
    name: str,
    workers: typing.List[behaviour.Behaviour],
    on_failure: behaviour.Behaviour,
    on_success: behaviour.Behaviour,
) -> behaviour.Behaviour:
    """
    Implement a multi-tick, general purpose 'try-except-else'-like pattern.

    This is a swiss knife version of the eventually idiom
    that facilitates a multi-tick response for specialised
    handling work sequence's completion status.

    .. graphviz:: dot/eventually-swiss.dot

    Args:
        name: the name to use for the idiom root
        workers: the worker behaviours or subtrees
        on_success: the behaviour or subtree to tick on work success
        on_failure: the behaviour or subtree to tick on work failure

    Returns:
        :class:`~py_trees.behaviour.Behaviour`: the root of the oneshot subtree

    .. seealso:: :meth:`py_trees.idioms.eventually`, :ref:`py-trees-demo-eventually-swiss-program`
    """
    on_success_sequence = composites.Sequence(
        name="Work to Success", memory=True, children=workers + [on_success]
    )
    on_failure_sequence = composites.Sequence(
        name="On Failure",
        memory=True,
        children=[on_failure, behaviours.Failure(name="Failure")],
    )
    subtree_root = composites.Selector(
        name=name, memory=False, children=[on_success_sequence, on_failure_sequence]
    )
    return subtree_root