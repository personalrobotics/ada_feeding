#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines an abstract class (and defaults) for 
selecting an action based on a policy.
"""

# Standard imports
import os
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import numpy.typing as npt

# Local imports
from ada_feeding_msgs.srv import AcquisitionSelect, AcquisitionReport
from ada_feeding_actionselect import helpers


class Policy(ABC):
    """
    An interface to select an action based on input
    """

    def __init__(self, 
        context_dim: int, 
        posthoc_dim: int):
        """
        Default self properties
        """
        self.context_dim = context_dim
        self.posthoc_dim = posthoc_dim

    def validate(self):
        """
        Validate variable arguments.
        Raise an exception on failure.
        """
        pass

    @abstractmethod
    def choice(
        self,
        context: npt.NDArray,
        response: AcquisitionSelect.Response
    ) -> AcquisitionSelect.Response:
        """
        Execute the policy to choose an action distribution.

        Parameters
        ----------
        context: Of dimension (context_dim, )
        response: service response passed to the callback

        Returns
        -------
        The input response populated based on the policy.
        See AcquisitionSelect.srv
        """
        raise NotImplementedError("choice not implemented")

    @abstractmethod
    def update(
        self,
        posthoc: npt.NDArray,
        request: AcquisitionReport.Request,
        response: AcquisitionSelect.Response
    ) -> AcquisitionSelect.Response:
        """
        Update the policy based on the result of action
        execution and any posthoc context.

        Parameters
        ----------
        posthoc: Of dimension (posthoc_dim, )
        request: service request passed to the callback
        response: service response passed to the callback

        Returns
        -------
        The input response populated based on the policy.
        See AcquisitionReport.srv
        """
        raise NotImplementedError("update not implemented")

class ConstantPolicy(Policy):
    """
    Execute a constant policy
    """

    def __init__(self, 
        context_dim: int, 
        posthoc_dim: int, 
        library: str = "",
        index: int = 0):
        """
        Default self properties
        """
        super().__init__(context_dim, posthoc_dim)
        self.library_path = library
        self.index = index

    def validate(self):
        """
        Validate variable arguments.
        Raise an exception on failure.
        """
        self.library = helpers.get_action_policy(self.library_path)
        if self.index < 0 or self.index >= len(self.library):
            raise IndexError(f"Index {index} is out of bounds: [0, {len(self.library)}]")

    def choice(
        self,
        context: npt.NDArray,
        response: AcquisitionSelect.Response
    ) -> AcquisitionSelect.Response:
        """
        Execute the policy to choose an action distribution.

        Parameters
        ----------
        context: Of dimension (context_dim, )
        response: service response passed to the callback

        Returns
        -------
        The input response populated based on the policy.
        See AcquisitionSelect.srv
        """
        response.actions = [self.library[self.index]]
        response.probabilities = [1.0]
        return response

    def update(
        self,
        posthoc: npt.NDArray,
        request: AcquisitionReport.Request,
        response: AcquisitionSelect.Response
    ) -> AcquisitionSelect.Response:
        """
        Update the policy based on the result of action
        execution and any posthoc context.

        Parameters
        ----------
        posthoc: Of dimension (posthoc_dim, )
        request: service request passed to the callback
        response: service response passed to the callback

        Returns
        -------
        The input response populated based on the policy.
        See AcquisitionReport.srv
        """
        return response
