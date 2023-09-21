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
    Trigger the child on preemption, e.g., when :meth:`terminate` is
    called with status :data:`~py_trees.common.Status.INVALID`. The
    child can either be triggered: (a) for a single tick; or (b) until it
    reaches a status other than :data:`~py_trees.common.Status.RUNNING` or
    times out.

    Always return `update_status` and on :meth:`terminate` (with new_status
    :data:`~py_trees.common.Status.INVALID`), call the child's
    :meth:`~py_trees.behaviour.Behaviour.update` method.

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
        update_status: common.Status = common.Status.RUNNING,
        single_tick: bool = True,
        period_ms: int = 0,
        timeout: typing.Optional[float] = None,
    ):
        """
        Initialise with the standard decorator arguments.

        Args:
            name: the decorator name
            child: the child to be decorated
            update_status: the status to return on :meth:`update`
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
        self.update_status = update_status
        self.single_tick = single_tick
        self.period_ms = period_ms
        self.timeout = timeout

    def tick(self) -> typing.Iterator[behaviour.Behaviour]:
        """
        Bypass the child when ticking.

        Yields:
            a reference to itself
        """
        self.logger.debug(f"{self.__class__.__name__}.tick()")
        self.status = self.update()
        yield self

    def update(self) -> common.Status:
        """
        Return the constant status specified in the constructor.

        Returns:
            the behaviour's new status :class:`~py_trees.common.Status`
        """
        return self.update_status

    def terminate(self, new_status: common.Status) -> None:
        """Tick the child behaviour once."""
        self.logger.debug(
            f"{self.__class__.__name__}.terminate({self.status}->{new_status})"
            if self.status != new_status
            else f"{new_status}",
        )
        if new_status == common.Status.INVALID:
            terminate_start_s = time.time()
            # Tick the child once
            self.decorated.tick_once()
            # If specified, tick until the child reaches a non-RUNNING status
            if not self.single_tick:
                while self.decorated.status == common.Status.RUNNING and (
                    self.timeout is None
                    or time.time() - terminate_start_s < self.timeout
                ):
                    if self.period_ms > 0:
                        time.sleep(self.period_ms / 1000.0)
                    self.decorated.tick_once()
            # Do not need to stop the child here - this method
            # is only called by Decorator.stop() which will handle
            # that responsibility immediately after this method returns.
