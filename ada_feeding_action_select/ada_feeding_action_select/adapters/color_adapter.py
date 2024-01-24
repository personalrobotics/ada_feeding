#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SPANet context adapter.

"""

# Standard imports

# Third-party imports
import numpy as np
import numpy.typing as npt
from overrides import override

# Local imports
from ada_feeding_action_select.helpers import logger
from ada_feeding_msgs.msg import Mask
from ada_feeding_perception.helpers import ros_msg_to_cv2_image
from .base_adapters import ContextAdapter


class ColorContext(ContextAdapter):
    """
    An adapter that gets some aggregate
    color from the mask.
    """

    def __init__(
        self,
        method: str = "mean",
    ) -> None:
        """
        Load Checkpoint and Set Config Parameters

        Parameters
        ----------
        method: how to aggregate the color, current options: "mean", "median"
        """
        self.method = method

    @property
    @override
    def dim(self) -> int:
        # Docstring copied from @override
        # Just RGB
        return 3

    @override
    def get_context(self, mask: Mask) -> npt.NDArray:
        # Docstring copied from @override

        # Prepare image(s)
        colors_masked = ros_msg_to_cv2_image(mask.rgb_image)[
            :, ros_msg_to_cv2_image(mask.mask) > 0
        ]

        # Aggregate
        if self.method == "median":
            ret = np.median(colors_masked, axis=1).flatten()
        else:
            ret = np.mean(colors_masked, axis=1).flatten()

        logger.info(f"Found Color: {ret}")
        assert ret.size == self.dim
        return ret
