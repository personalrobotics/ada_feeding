#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeFoodFrame behavior, which computes the
food frame from 
"""
# Standard imports

# Third-party imports
import py_trees

# Local imports


class ComputeFoodFrame(py_trees.behaviour.Behaviour):
    """
    TODO
    """

    def __init__(
        self,
        name: str,
        node: Node,
        food_detection_input_key: str,
        frame_output_key: str,
        world_frame_id: str,
    ) -> None:
        # Initiatilize the behavior
        super().__init__(name=name)
        pass

    def setup(self, **kwargs) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        pass

class FlipFoodFrame(py_trees.behavior.Behavior):
    """
    TODO
    Take the food frame from the blackboard and 
    rotate it PI about the world frame +Z axis.
    """

    def __init__(
        self,
        name: str,
        node: Node,
        frame_blackboard_key: str,
        world_frame_id: str,
    ) -> None:
        # Initiatilize the behavior
        super().__init__(name=name)
        pass

    def setup(self, **kwargs) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        pass
