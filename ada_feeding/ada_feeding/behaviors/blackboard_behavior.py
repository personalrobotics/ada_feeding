#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic behavior with additional functionality for blackboard
remappings (inspired by BehaviorTree.CPP)
"""

# Standard imports
from typing import Any, Dict, Optional, Union

# Third-party imports
import py_trees

# Local imports
from ada_feeding.helpers import BlackboardKey


class BlackboardBehavior(py_trees.behaviour.Behaviour):
    """
    A super-class for any py_tree Behaviour that adds
    an easy interface for working with the internal blackboard.

    _remap, _inputs, _outputs should not be overridden.
    self.blackboard can be used to get Client information,
    but in general reads/writes should go through the function API.
    """

    def __init__(self, 
        name: str, 
        ns: str = "/",
        inputs: Optional[Dict[str, Union[BlackboardKey, Any]]] = None,
        outputs: Optional[Dict[str, Optional[BlackboardKey]]] = None) -> None:
        """
        Initialize the behavior and blackboard

        Parameters
        ----------
        name: Behavior name
        ns: Blackboard namespace (usually the name of the tree / subtree)
        inputs: optional kwargs for call to self.blackboard_inputs
        outputs: optional kwargs for call to self.blackboard_outputs
        """
        super().__init__(name=name)
        self.blackboard = super().attach_blackboard_client(name=name, namespace=ns)
        self._remap = {}
        self._inputs = {}
        self._outputs = {}
        if inputs is not None:
            self.blackboard_inputs(**inputs)
        if outputs is not None:
            self.blackboard_outputs(**outputs)

    def blackboard_inputs(self, **kwargs) -> None:
        """
        Define and register all blackboard input keys.

        Each key of the kwargs is a str specifying the location on the blackboard 
        for that variable to be stored. Each value of kwargs is either `BlackboardKey`,
        in which case it represents a blackboard location to remap the key to, 
        or another type, in which case it represents a constant value to store at that key. 
        Note that as opposed to setting constants on the blackboard, 
        this behavior stores it in a local dict.

        Generally this is done in the subclass as follows:
        <key>: Union[BlackboardKey, <type>] = <default>

        Call from the subclass using:
        super().blackboard_inputs(**{key: value for key,
            value in locals().items() if key != 'self'})

        Constants will be stored in the local dict.
        """
        for key, value in kwargs.items():
            if isinstance(value, BlackboardKey):
                self.blackboard.register_key(
                    key=value, access=py_trees.common.Access.READ
                )
                self._remap[key] = value
            else:
                self._inputs[key] = value

    def blackboard_outputs(self, **kwargs) -> None:
        """
        Define and register all blackboard output keys.

        As in blackboard_inputs, kwargs keys are strings passed
        to calls to `blackboard_set`, while values are the
        corresponding blackboard keys (or None to disable writing).

        Generally this is done in the subclass as follows:
        <key>: Optional[BlackboardKey] = <default>

        Call from the subclass using:
        super().blackboard_outputs(**{key: value for key,
            value in locals().items() if key != 'self'})
        """
        for key, value in kwargs.items():
            if isinstance(value, BlackboardKey):  # Catches None
                self.blackboard.register_key(
                    key=value, access=py_trees.common.Access.WRITE
                )
            self._outputs[key] = value

    def blackboard_exists(self, key: str) -> bool:
        """
        Check if a key is set in the blackboard or available locally.
        Raises KeyError if the key has not been defined in blackboard_inputs
        """
        if key in self._inputs:
            return True

        return self.blackboard.exists(self._remap[key])

    def blackboard_get(self, key: str) -> Any:
        """
        Return a key from either the blackboard or local store.
        Raises KeyError if the key has not been defined in blackboard_inputs
        """
        if key in self._inputs:
            return self._inputs[key]

        return self.blackboard.get(self._remap[key])

    def blackboard_set(self, key: str, output: Any) -> None:
        """
        Write a key to the blackboard
        Raises KeyError if the key has not been defined in blackboard_outputs
        """

        if key not in self._outputs:
            raise KeyError(f"{key} is not a registered output: {self._outputs.keys()}")

        if self._outputs[key] is not None:
            self.blackboard.set(self._outputs[key], output)
        else:
            self.get_logger().debug(f"Not writing to 'None' output key: {self._outputs[key]}")
