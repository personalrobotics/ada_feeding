"""
NOTE: This is a multi-tick version of the decorator discussed in
https://github.com/splintered-reality/py_trees/pull/427 . Once a
multi-tick version of that decorator is merged into py_trees, this
decorator should be removed in favor of the main py_trees one.
"""

import functools
import inspect
import time
import typing

from py_trees import behaviour, blackboard, common
from py_trees.decorators import Decorator

class OnTerminate(Decorator):
    """
    Trigger the child for a single tick on :meth:`terminate`.

    Always return :data:`~py_trees.common.Status.RUNNING` and on
    on :meth:`terminate`, call the child's
    :meth:`~py_trees.behaviour.Behaviour.update` method, once.

    This is useful to cleanup, restore a context switch or to
    implement a finally-like behaviour.

    .. seealso:: :meth:`py_trees.idioms.eventually`
    """

    def __init__(self, name: str, child: behaviour.Behaviour):
        """
        Initialise with the standard decorator arguments.

        Args:
            name: the decorator name
            child: the child to be decorated
        """
        super(OnTerminate, self).__init__(name=name, child=child)

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
        Return with :data:`~py_trees.common.Status.RUNNING`.

        Returns:
            the behaviour's new status :class:`~py_trees.common.Status`
        """
        return common.Status.RUNNING

    def terminate(self, new_status: common.Status) -> None:
        """Tick the child behaviour once."""
        self.logger.debug(
            "{}.terminate({})".format(
                self.__class__.__name__,
                "{}->{}".format(self.status, new_status)
                if self.status != new_status
                else f"{new_status}",
            )
        )
        if new_status == common.Status.INVALID:
            self.decorated.tick_once()
            # Do not need to stop the child here - this method
            # is only called by Decorator.stop() which will handle
            # that responsibility immediately after this method returns.