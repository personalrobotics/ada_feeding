#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a policy that chooses an action
based on the color of the mask.
"""

# Standard imports
import os
from typing import List, Tuple, Union
import yaml

# Third-party imports
from ament_index_python.packages import get_package_share_directory
from overrides import override
import numpy as np
import numpy.typing as npt

# Local imports
from ada_feeding_msgs.msg import AcquisitionSchema
from ada_feeding_action_select.helpers import get_action_library, logger
from .base_policies import Policy


class ColorPolicy(Policy):
    """
    Do an action based on color.
    """

    def __init__(
        self, context_dim: int, posthoc_dim: int, library: str = "", color_map: str = ""
    ):
        """
        Default self properties
        """
        super().__init__(context_dim, posthoc_dim)
        if context_dim != 3:
            raise ValueError("Must be a 3D RGB context.")
        self.library = get_action_library(library)
        self.colors, self.indices = self.get_color_map(color_map)
        for index in self.indices:
            if index < 0 or index >= len(self.library):
                raise IndexError(
                    f"Index {index} from color map is out of bounds: [0, {len(self.library)}]"
                )

    def get_color_map(
        self, map_path: str, key: str = "color_map"
    ) -> Tuple[List[npt.NDArray], List[int]]:
        """
        Loads library for actions from a YAML file.

        Parameters
        ----------
        map_path: location of yaml relative to package share
        key: Must be an array of color: and index: pairs

        Returns
        -------
        Parallel arrays of Colors and action indices
        """

        package_share = get_package_share_directory("ada_feeding_action_select")
        file_name = os.path.join(package_share, map_path)

        try:
            with open(file_name, "r", encoding="utf-8") as file:
                yaml_file = yaml.safe_load(file)
        except Exception as exception:
            raise exception.__class__(
                f"Failed to load color map at {file_name}. "
                f"Are you passing in the right color_map parameter value? {exception}"
            )

        colors = []
        indices = []
        for element in yaml_file[key]:
            color = np.array(element["color"]).flatten()
            assert color.size == self.context_dim
            colors.append(color)
            indices.append(int(element["index"]))

        return colors, indices

    @override
    def choice(
        self, context: npt.NDArray
    ) -> Union[List[Tuple[float, AcquisitionSchema]], str]:
        # Docstring copied from @override

        min_color_dist = np.inf
        best_color = np.zeros(3)
        best_index = 0
        for index, color in enumerate(self.colors):
            vec = context - color
            color_dist = np.dot(vec, vec)
            if color_dist < min_color_dist:
                min_color_dist = color_dist
                best_index = self.indices[index]

        logger.info(f"Closest color is {best_color}, returning {best_index}")
        return [(1.0, self.library[best_index])]

    def update(
        self,
        posthoc: npt.NDArray,
        context: npt.NDArray,
        action: Tuple[float, AcquisitionSchema],
        loss: float,
    ) -> Tuple[bool, str]:
        # Docstring copied from @override
        return (True, "Success")
