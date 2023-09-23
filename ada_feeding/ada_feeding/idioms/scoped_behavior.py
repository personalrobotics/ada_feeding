"""
This module defines the `scoped_behavior` idiom, which is a way to run a main
behavior within the scope of a pre and post behavior.

In expected usage, the pre behavior will open or create a resources, the main
behavior will use those resources, and the post behavior will close or delete the
resources. The idiom guarentees the following:
    1. The main behavior will not be ticked unless the pre behavior returns
       SUCCESS.
    2. The behavior returned by this idiom will not reach a terminal (non-RUNNING)
       status until the post behavior has been ticked to a terminal status. In
       other words, regardless of whether the main behavior returns SUCCESS,
       FAILURE, or if the idiom is preempted (e.g., had `stop(INVALID)` called
       on it), the post behavior will still be ticked till a terminal status.
    3. The root behavior's terminal status will be FAILURE if the pre behavior
       returns FAILURE, else it will be the main behavior's terminal status.

Note the following nuances:
    1. If the main behaviour reaches SUCCESS or FAILURE, the post behaviour will
       be ticked asynchronously during the standard `tick()` of the tree. However,
       if the idiom is preempted, the post behaviour will be ticked synchronously,
       as part of the `stop(INVALID)` code of the tree, e.g., progression of
       the `stop(INVALID)` code will be blocked until the post behaviour reaches
       a terminal status.
    2. It is possible that the post behavior will be ticked to completion multiple
       times. For example, consider the case where the main behavior succeeds,
       the post behavior succeeds, and then the idiom is preempted. Therefore,
       the post behavior should be designed in a way that it can be run to completion
       multiple times sequentially, without negative side effects.
"""

# Standard imports
from typing import List, Optional

# Third-party imports
import py_trees
from py_trees.behaviours import BlackboardToStatus, UnsetBlackboardVariable
from py_trees.decorators import (
    FailureIsSuccess,
    StatusToBlackboard,
)

# Local imports
from ada_feeding.decorators import OnPreempt


# pylint: disable=too-many-arguments
# One over is fine.
def scoped_behavior(
    name: str,
    pre_behavior: py_trees.behaviour.Behaviour,
    workers: List[py_trees.behaviour.Behaviour],
    post_behavior: py_trees.behaviour.Behaviour,
    on_preempt_period_ms: int = 0,
    on_preempt_timeout: Optional[float] = None,
    status_blackboard_key: Optional[str] = None,
) -> py_trees.behaviour.Behaviour:
    """
    Returns a behavior that runs the main behavior within the scope of the pre
    and post behaviors. See the module docstring for more details.

    Parameters
    ----------
    name: The name to associate with this behavior.
    pre_behavior: The behavior to run before the main behavior.
    workers: The behaviors to run in the middle.
    post_behavior: The behavior to run after the main behavior.
    on_preempt_period_ms: How long to sleep between ticks (in milliseconds)
        if the behavior gets preempted. If 0, then do not sleep.
    on_preempt_timeout: How long (sec) to wait for the behavior to reach a
        terminal status if the behavior gets preempted. If None, then do not
        timeout.
    status_blackboard_key: The blackboard key to use to store the status of
        the behavior. If None, use `/{name}/scoped_behavior_status`.
    """
    if status_blackboard_key is None:
        status_blackboard_key = f"/{name}/scoped_behavior_status"

    main_sequence = py_trees.composites.Sequence(
        name="Scoped Behavior",
        memory=True,
    )

    # First, unset the status variable.
    unset_status = UnsetBlackboardVariable(
        name="Unset Status", key=status_blackboard_key
    )
    main_sequence.children.append(unset_status)

    # Then, execute the pre behavior and the workers
    pre_and_workers_sequence = py_trees.composites.Sequence(
        name="Pre & Workers",
        children=[pre_behavior] + workers,
        memory=True,
    )
    write_workers_status = StatusToBlackboard(
        name="Write Pre & Workers Status",
        child=pre_and_workers_sequence,
        variable_name=status_blackboard_key,
    )
    workers_branch = FailureIsSuccess(
        name="Pre & Workers Branch",
        child=write_workers_status,
    )
    main_sequence.children.append(workers_branch)

    # Then, execute the post behavior
    post_branch = FailureIsSuccess(
        name="Post Branch",
        child=post_behavior,
    )
    main_sequence.children.append(post_branch)

    # Finally, write the status of the main behavior to the blackboard.
    write_status = BlackboardToStatus(
        name="Write Status",
        variable_name=status_blackboard_key,
    )
    main_sequence.children.append(write_status)

    # To handle preemptions, we place the main behavior into an OnPreempt
    # decorator, with `post` as the preemption behavior.
    root = OnPreempt(
        name=name,
        child=main_sequence,
        on_preempt=post_behavior,
        single_tick=False,
        period_ms=on_preempt_period_ms,
        timeout=on_preempt_timeout,
    )

    return root
