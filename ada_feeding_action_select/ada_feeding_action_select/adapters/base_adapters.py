#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines an abstract class (and defaults) for generating
context and posthoc vectors from input data.
"""

# Standard imports
from abc import ABC, abstractmethod
from typing import Optional

# Third-party imports
import numpy as np
import numpy.typing as npt

# Local imports
from ada_feeding_msgs.msg import Mask


class ContextAdapter(ABC):
    """
    An interface to translate a visual Mask to a context vector.
    """

    def __init__(self):
        """
        Default self properties
        """

        # These attributes are used by the policy service
        # To determine whether to pass image/depth data
        # to get_context.
        self.need_rgb = False
        self.need_depth = False

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Property defining the size of the context vector
        """
        raise NotImplementedError("dimension not implemented")

    @abstractmethod
    def get_context(
        self, mask: Mask, image: Optional[npt.NDArray], depth: Optional[npt.NDArray]
    ) -> npt.NDArray:
        """
        Create the context vector from the provided visual info

        Parameters
        ----------
        mask: See Mask.msg
        image: Full camera image, None if self.need_rgb is False
        image: Full depth image, None if self.need_depth is False

        Returns
        -------
        A flat numpy array, shape (dim,)
        """
        raise NotImplementedError("get_context not implemented")


class PosthocAdapter(ABC):
    """
    An interface to translate any input info to a posthoc vector.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Property defining the size of the posthoc vector
        """
        raise NotImplementedError("dimension not implemented")

    @abstractmethod
    def get_posthoc(self, data: npt.NDArray) -> npt.NDArray:
        """
        Create the posthoc vector from the provided info

        Parameters
        ----------
        data: AcquisitionReport.posthoc

        Returns
        -------
        A flat numpy array, shape (dim,)
        """
        raise NotImplementedError("get_posthoc not implemented")


class NoContext(ContextAdapter, PosthocAdapter):
    """
    An instantiation that just returns [0] for context and posthoc.
    """

    @property
    def dim(self) -> int:
        return 1

    def get_context(
        self, mask: Mask, image: Optional[npt.NDArray], depth: Optional[npt.NDArray]
    ) -> npt.NDArray:
        return np.array([0.0])

    def get_posthoc(self, data: npt.NDArray) -> npt.NDArray:
        return np.array([0.0])
