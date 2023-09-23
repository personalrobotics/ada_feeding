"""
NOTE: This is a multi-tick version of the decorator discussed in
https://github.com/splintered-reality/py_trees/pull/427 . Once a
multi-tick version of that decorator is merged into py_trees, this
decorator should be removed in favor of the main py_trees one.
"""

import time
import typing

from py_trees import behaviour, common
from py_trees.decorators import Decorator


class OnPreempt(Decorator):
    """
    Behaves identically to :class:`~py_trees.decorators.PassThrough` except
    that if it gets preempted (i.e., `terminate(INVALID)` is called on it)
    while its status is :data:`~py_trees.common.Status.RUNNING`, it will
    tick `on_preempt` either: (a) for a single tick; or (b) until `on_preempt`
    reaches a status other than :data:`~py_trees.common.Status.RUNNING` or
    times out. Note that `on_preempt` may be a behavior that exists elsewhere
    in the tree, or it may be a separate behavior.

    This is useful to cleanup, restore a context switch or to
    implement a finally-like behaviour.

    .. seealso:: :meth:`py_trees.idioms.eventually`, :meth:`py_trees.idioms.eventually_swiss`
    """

    # pylint: disable=too-many-arguments
    # This is acceptable, to give users maximum control over how this decorator
    # behaves.
    def __init__(
        self,
        name: str,
        child: behaviour.Behaviour,
        on_preempt: behaviour.Behaviour,
        single_tick: bool = True,
        period_ms: int = 0,
        timeout: typing.Optional[float] = None,
    ):
        """
        Initialise with the standard decorator arguments.

        Args:
            name: the decorator name
            child: the child to be decorated
            on_preempt: the behaviour or subtree to tick on preemption
            single_tick: if True, tick the child once on preemption. Else,
                tick the child until it reaches a status other than
                :data:`~py_trees.common.Status.RUNNING`.
            period_ms: how long to sleep between ticks (in milliseconds)
                if `single_tick` is False. If 0, then do not sleep.
            timeout: how long (sec) to wait for the child to reach a status
                other than :data:`~py_trees.common.Status.RUNNING` if
                `single_tick` is False. If None, then do not timeout.
        """
        super().__init__(name=name, child=child)
        self.on_preempt = on_preempt
        self.single_tick = single_tick
        self.period_ms = period_ms
        self.timeout = timeout

    def update(self) -> common.Status:
        """
        Just reflect the child status.

        Returns:
            the behaviour's new status :class:`~py_trees.common.Status`
        """
        return self.decorated.status

    def stop(self, new_status: common.Status) -> None:
        """
        Check if the child is running (dangling) and stop it if that is the case.

        This function departs from the standard :meth:`~py_trees.decorators.Decorator.stop`
        in that it *first* stops the child, and *then* stops the decorator.

        Args:
            new_status (:class:`~py_trees.common.Status`): the behaviour is transitioning
            to this new status
        """
        self.logger.debug(f"{self.__class__.__name__}.stop({new_status})")
        # priority interrupt handling
        if new_status == common.Status.INVALID:
            self.decorated.stop(new_status)
        # if the decorator returns SUCCESS/FAILURE and should stop the child
        if self.decorated.status == common.Status.RUNNING:
            self.decorated.stop(common.Status.INVALID)
        self.terminate(new_status)
        self.status = new_status

    def terminate(self, new_status: common.Status) -> None:
        """Tick the child behaviour once."""
        self.logger.debug(
            f"{self.__class__.__name__}.terminate({self.status}->{new_status})"
        )
        if new_status == common.Status.INVALID and self.status == common.Status.RUNNING:
            terminate_start_s = time.monotonic()
            # Tick the child once
            self.on_preempt.tick_once()
            # If specified, tick until the child reaches a non-RUNNING status
            if not self.single_tick:
                while self.on_preempt.status == common.Status.RUNNING and (
                    self.timeout is None
                    or time.monotonic() - terminate_start_s < self.timeout
                ):
                    if self.period_ms > 0:
                        time.sleep(self.period_ms / 1000.0)
                    self.on_preempt.tick_once()
            # Do not need to stop the child here - this method
            # is only called by Decorator.stop() which will handle
            # that responsibility immediately after this method returns.
