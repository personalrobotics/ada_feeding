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
from typing import Callable, List, Optional

# Third-party imports
import py_trees

# Local imports
from .eventually_swiss import eventually_swiss


# pylint: disable=too-many-arguments
# One over is fine.
def scoped_behavior(
    name: str,
    pre_behavior: py_trees.behaviour.Behaviour,
    main_behaviors: List[py_trees.behaviour.Behaviour],
    post_behavior_fn: Callable[[], py_trees.behaviour.Behaviour],
    on_preempt_period_ms: int = 0,
    on_preempt_timeout: Optional[float] = None,
) -> py_trees.behaviour.Behaviour:
    """
    Returns a behavior that runs the main behavior within the scope of the pre
    and post behaviors. See the module docstring for more details.

    Parameters
    ----------
    name: The name to associate with this behavior.
    pre_behavior: The behavior to run before the main behavior.
    main_behaviors: The behaviors to run in the middle.
    post_behavior_fn: A function that returns the behavior to run after the main
        behavior. This must be a function because the post behavior will be
        reused at multiple locations in the tree.
    on_preempt_period_ms: How long to sleep between ticks (in milliseconds)
        if the behavior gets preempted. If 0, then do not sleep.
    on_preempt_timeout: How long (sec) to wait for the behavior to reach a
        terminal status if the behavior gets preempted. If None, then do not
        timeout.
    """
    return eventually_swiss(
        name=name,
        workers=[pre_behavior] + main_behaviors,
        on_failure=post_behavior_fn(),
        on_success=post_behavior_fn(),
        on_preempt=post_behavior_fn(),
        on_preempt_single_tick=False,
        on_preempt_period_ms=on_preempt_period_ms,
        on_preempt_timeout=on_preempt_timeout,
    )
