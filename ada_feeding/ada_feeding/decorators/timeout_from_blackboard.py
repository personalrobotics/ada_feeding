"""
This module extends the Timeout decorator
to accept a blackboard namespace and 
"""

# Standard imports

# Third-party imports
from overrides import override
from py_trees.decorators import Timeout
import py_trees

# Local imports
from ada_feeding.helpers import BlackboardKey


class TimeoutFromBlackboard(Timeout):
    """
    A decorator that has the same functionality
    as the parent class, but allows the passing
    in of a blackboard key for the duration
    parameter.

    """
    def __init__(
        self,
        name: str,
        ns: str,
        duration_key: BlackboardKey,
        child: py_trees.behaviour.Behaviour,
    ) -> None:
        """
        Call parent w/ name and child
        Just store local blackboard client + duration
        """
        self._blackboard = py_trees.blackboard.Client(name=name, namespace=ns)
        self._blackboard.register_key(key=duration_key, access=py_trees.common.Access.READ)
        self._duration_key = duration_key

        super().__init__(name=name, child=child)

    @override
    def initialise(self):
        self.duration = self._blackboard.get(self._duration_key)
        super().initialise()
