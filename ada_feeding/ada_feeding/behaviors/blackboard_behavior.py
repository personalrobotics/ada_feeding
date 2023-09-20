#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic behavior with additional functionality for blackboard
remappings (inspired by BehaviorTree.CPP)
"""

# Standard imports
from typing import Any

# Third-party imports
import py_trees

# Local imports
from ada_feeding.helpers import BlackboardKey

class BlackboardBehavior(py_trees.behaviour.Behaviour):
    """
    TODO
    """

    def __init__(
        self,
        ns: str = "/",
        name: str
    ) -> None:
        """
        Initialize the behavior and blackboard

        Parameters
        ----------
        ns: Blackboard namespace (usually the name of the tree / subtree)
        name: Behavior name
        """
        super().__init__(name=name)
        self.blackboard = py_trees.blackboard.Client(name=name, namespace=ns)
        self.keys = dict()
        self.locals = dict()
        self.outputs = dict()

    def blackboard_inputs(
        self, **kwargs
    ) -> None:
        """
        Define and register all blackboard input keys.

        Generally this is done in the subclass as follows:
        <key>: Union[BlackboardKey, <type>] = <default>

        Call from the subclass using:
        super().blackboard_inputs(**{key: value for key, value in locals().items() if key != 'self'})

        Constants will be stored in the local dict.
        """
        for key, value in kwargs.items():
            if isinstance(value, BlackboardKey):
                self.blackboard.register_key(key=value, access=py_trees.common.Access.READ)
                self.keys[key] = value
            else:
                self.locals[key] = value

    def blackboard_outputs(
        self, **kwargs
    ) -> None:
        """
        Define and register all blackboard output keys.

        Generally this is done in the subclass as follows:
        <key>: BlackboardKey = <default>

        Call from the subclass using:
        super().blackboard_inputs(**{key: value for key, value in locals().items() if key != 'self'})
        """
        for key, value in kwargs.items():
            if value is not None:
                self.blackboard.register_key(key=value, access=py_trees.common.Access.WRITE)
            self.outputs[key] = value

    def blackboard_exists(
        self, key: str
    ) -> bool:
        """
        Check if a key is set in the blackboard or available locally.
        """
        if key in self.locals.keys():
            return True

        return self.blackboard.exists(self.keys[key])

    def blackboard_get(
        self, key: str
    ) -> Any:
        """
        Return a key from either the blackboard or local store.
        """
        if key in self.locals.keys():
            return self.locals[key]

        return self.blackboard.get(self.keys[key])

    def blackboard_write(
        self, key: str, output: Any
    ) -> None:
        """
        Write a key to the blackboard
        """

        if key not in self.outputs.keys():
            raise KeyError(f"{key} is not a registered output: {self.outputs.keys()}")

        if self.outputs[key] is not None:
            self.blackboard.set(self.outputs.keys[key], output)
