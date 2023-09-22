"""
This module defines the ForceStatus decorator, which ignores the child's
current status and always returns the status specified by the user.
"""

# Standard imports
import typing

# Third-party imports
from py_trees import behaviour, common
from py_trees.decorators import Decorator

# Local imports


class ForceStatus(Decorator):
    """
    This decorator ignores the child's current status and always returns
    the status specified by the user. This can be useful e.g., to force a
    composite without memory to move on to the next child while still ticking
    the child of this decorator.

    While this decorator's behavior can be achieved with a chain of "X is Y"
    style decorators, it is often conceptually easier to reason about a decorator
    always returning one status, as opposed to a chain of decorators that each
    flip their child's status.
    """

    def __init__(self, name: str, child: behaviour.Behaviour, status: common.Status):
        """
        Initialise the decorator.

        Args:
            name: the decorator name
            child: the child to be decorated
            status: the status to return on :meth:`update`
        """
        super().__init__(name=name, child=child)
        self.const_status = status

    def tick(self) -> typing.Iterator[behaviour.Behaviour]:
        """
        Don't stop the child even if this decorator's status is non-RUNNING.

        Yields:
            a reference to itself
        """
        self.logger.debug(f"{self.__class__.__name__}.tick()")
        # initialise just like other behaviours/composites
        if self.status != common.Status.RUNNING:
            self.initialise()
        # interrupt proceedings and process the child node
        # (including any children it may have as well)
        for node in self.decorated.tick():
            yield node
        # resume normal proceedings for a Behaviour's tick
        self.status = self.update()
        # do not stop even if this decorator's status is non-RUNNING
        yield self

    def update(self) -> common.Status:
        """
        Return the status specified when creating this decorator.

        Returns:
            the behaviour's new status :class:`~py_trees.common.Status`
        """
        return self.const_status
