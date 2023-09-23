#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the TODO
"""
# Standard imports

# Third-party imports
import py_trees

# Local imports


class ComputeMoveOut(py_trees.behaviour.Behaviour):
    """
    TODO
    """

    def __init__(
        self,
        name: str,
        node: Node,
    ) -> None:
        # Initiatilize the behavior
        super().__init__(name=name)
        pass

    def setup(self, **kwargs) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        pass
