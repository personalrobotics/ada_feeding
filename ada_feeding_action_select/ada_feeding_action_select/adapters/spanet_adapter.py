#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SPANet context adapter.

"""

# Standard imports
from abc import ABC, abstractmethod
from typing import Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
from overrides import override

# Local imports
from ada_feeding_msgs.msg import Mask


class SPANetAdapter(ContextAdapter):
    """
    An ad
    """

    @property
    def dim(self) -> int:
        return 1

    def get_context(self, mask: Mask) -> npt.NDArray:
        return np.array([0.0])

    def get_posthoc(self, data: npt.NDArray) -> npt.NDArray:
        return np.array([0.0])
