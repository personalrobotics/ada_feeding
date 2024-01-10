#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines an abstract class (and defaults) for 
selecting an action based on a policy.
"""

# Standard imports
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple

# Third-party imports
from overrides import override
import numpy.typing as npt

# Local imports
from ada_feeding_msgs.msg import AcquisitionSchema
from ada_feeding_action_select import helpers


class Policy(ABC):
    """
    An interface to select an action based on input
    """

    def __init__(self, context_dim: int, posthoc_dim: int):
        """
        Default self properties

        Parameters
        ----------
        context_dim: Provided context vectors should be of shape (context_dim, )
        posthod_cim: Provided posthoc vectors should be of shape (posthoc_dim, )
        """
        self.context_dim = context_dim
        self.posthoc_dim = posthoc_dim

    def save_checkpoint(self) -> Any:
        """
        Checkpoint the current policy and return it.
        This will be saved to a file by the policy service
        using torch.save({"checkpoint": <returned object>}).

        Returns
        -------
        Arbitrary serializeable object that checkpoints the policy.
        Return None if policy does not support checkpoints.
        """
        return None

    def load_checkpoint(self, checkpoint: Any) -> bool:
        """
        Load a checkpoint from a provided serializeable
        object. This should ideally be the same type of
        object as returned by save_checkpoint().

        Returns
        -------
        Whether checkpoint loads successfully or not.
        """
        # pylint: disable=unused-argument
        # Abstract method
        return True

    @abstractmethod
    def choice(
        self, context: npt.NDArray
    ) -> Union[Dict[float, AcquisitionSchema], str]:
        """
        Execute the policy to choose an action distribution.

        Parameters
        ----------
        context: Of dimension (context_dim, )

        Returns
        -------
        Either an action distribution as a dictionary: [probability: action]
        OR a status string on error.
        """
        raise NotImplementedError("choice not implemented")

    @abstractmethod
    def update(
        self,
        posthoc: npt.NDArray,
        context: npt.NDArray,
        action: Tuple[float, AcquisitionSchema],
        loss: float,
    ) -> Tuple[bool, str]:
        """
        Update the policy based on the result of action
        execution and any posthoc context.

        Parameters
        ----------
        posthoc: Of dimension (posthoc_dim, )
        context: Of dimension (context_dim, )
        action: action taken by the agent and its probability
        loss: loss incurred by the agent

        Returns
        -------
        True/False on success/failure
        Status message
        """
        raise NotImplementedError("update not implemented")


class ConstantPolicy(Policy):
    """
    Execute a constant policy
    """

    def __init__(
        self, context_dim: int, posthoc_dim: int, library: str = "", index: int = 0
    ):
        """
        Default self properties
        """
        super().__init__(context_dim, posthoc_dim)
        self.index = index
        self.library = helpers.get_action_library(library)
        if self.index < 0 or self.index >= len(self.library):
            raise IndexError(
                f"Index {index} is out of bounds: [0, {len(self.library)}]"
            )

    @override
    def choice(
        self, context: npt.NDArray
    ) -> Union[Dict[float, AcquisitionSchema], str]:
        # Docstring copied from @override
        return {1.0: self.library[self.index]}

    def update(
        self,
        posthoc: npt.NDArray,
        context: npt.NDArray,
        action: Tuple[float, AcquisitionSchema],
        loss: float,
    ) -> Tuple[bool, str]:
        # Docstring copied from @override
        return (True, "Success")
