#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines an abstract class (and defaults) for 
selecting an action based on a policy.
"""
# pylint: disable=too-many-instance-attributes
# This is a dataclass, come on.

# Standard imports
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

# Third-party imports
from overrides import override
import numpy as np
import numpy.typing as npt

# Local imports
from ada_feeding_action_select.helpers import get_action_library, logger
from ada_feeding_msgs.msg import AcquisitionSchema
from .base_policies import Policy


@dataclass
class LinearPolicyCheckpoint:
    """
    Data Class for linear checkpoint parameters
    """

    # pylint: disable=too-many-instance-attributes
    # This is a dataclass, come on.

    # Required Equal Parameters
    # Must equal loaded or fail
    n_actions: int
    context_dim: int
    posthoc_dim: int

    # Overwritten Parameters
    # Will not be loaded
    library: List[AcquisitionSchema]
    lambda_l2: float
    lambda_posthoc_damp: float
    use_posthoc: bool
    linear_model: npt.NDArray

    # Variable Parameters
    # Will be loaded
    a_context: npt.NDArray
    b_context: npt.NDArray
    a_posthoc: npt.NDArray
    b_posthoc: npt.NDArray
    a_context_posthoc: npt.NDArray


class LinearPolicy(Policy):
    """
    Execute a linear policy on a discrete
    set of actions.
    Assumption:
    loss in R^n_actions = A*Context = B*Posthoc
    Solve Least Squares:
    (f_context - loss)^2 + (f_posthoc - loss)^2,
    s.t. f_context = f_posthoc
    """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        context_dim: int,
        posthoc_dim: int,
        library: str = "",
        lambda_l2=1e0,
        lambda_posthoc_damp=1e0,
        use_posthoc=False,
    ):
        """
        Define self properties
        """
        super().__init__(context_dim, posthoc_dim)
        self.library = get_action_library(library)

        # Cache inverses in case a subclass needs them
        self.cache = {}

        self.checkpoint = LinearPolicyCheckpoint(
            context_dim=context_dim,
            posthoc_dim=posthoc_dim,
            n_actions=self.n_actions,
            library=self.library,
            lambda_l2=lambda_l2,
            lambda_posthoc_damp=lambda_posthoc_damp,
            use_posthoc=use_posthoc,
            ### Context Linear Regression Params ###
            # Context Policy Vector
            linear_model=np.zeros((self.n_actions, self.context_dim)),
            # Context Per-Action Data Matrtix
            a_context=np.zeros((self.n_actions, self.context_dim, self.context_dim)),
            # Context Data-Loss Vector
            b_context=np.zeros((self.n_actions, self.context_dim)),
            ### Posthoc Linear Regression Params ###
            # Posthoc Per-Action Data Matrix
            a_posthoc=np.zeros((self.n_actions, self.posthoc_dim, self.posthoc_dim)),
            # Posthoc Data-Loss Vector
            b_posthoc=np.zeros((self.n_actions, self.posthoc_dim)),
            # Context/Posthoc Cross Data Matrix
            a_context_posthoc=np.zeros((self.context_dim, self.posthoc_dim)),
        )

    @property
    def n_actions(self) -> int:
        """
        Number of discrete actions
        """
        return len(self.library)

    @override
    def choice(
        self, context: npt.NDArray
    ) -> Union[List[Tuple[float, AcquisitionSchema]], str]:
        # Docstring copied from @override

        # Default to Random Policy
        # pylint: disable=unused-argument
        return [(1.0 / self.n_actions, action) for action in self.library]

    @override
    def update(
        self,
        posthoc: npt.NDArray,
        context: npt.NDArray,
        action: Tuple[float, AcquisitionSchema],
        loss: float,
    ) -> Tuple[bool, str]:
        # Docstring copied from @override
        prob = action[0]
        if prob <= 0.0:
            return (False, "0 Probability in linear policy update")

        try:
            arm = self.library.index(action[1])
        except ValueError:
            return (False, "Action not in linear policy library")

        # Record context
        outer = np.outer(context, context) / prob
        self.checkpoint.a_context[arm, :] += outer
        self.checkpoint.b_context[arm, :] += loss * context / prob

        # Record Posthoc
        outer = np.outer(posthoc, posthoc) / prob
        self.checkpoint.a_posthoc[arm, :] += outer
        self.checkpoint.b_posthoc[arm, :] += loss * posthoc / prob

        # Record Cross Params
        posthoc_context = np.outer(posthoc, context)
        assert posthoc_context.T.shape == (self.context_dim, self.posthoc_dim)
        self.checkpoint.a_context_posthoc += posthoc_context.T

        # Update Policy
        self._solve()
        return (True, "Success")

    def _solve(self):
        """
        Solve for policy from data matrices
        """

        # Context Only
        # Add L2 Regularization
        solve_a = self.checkpoint.a_context + (
            self.checkpoint.lambda_l2 * np.eye(self.context_dim)
        )
        solve_b = self.checkpoint.b_context

        # Posthoc Augmented
        if self.checkpoint.use_posthoc:
            # Add Posthoc Dampening Term
            a_posthoc = (
                self.checkpoint.a_posthoc
                + self.checkpoint.lambda_posthoc_damp * np.eye(self.posthoc_dim)
            )
            a_posthoc_inv = np.linalg.inv(np.sum(a_posthoc, axis=0))
            solve_a = (
                solve_a
                + self.checkpoint.a_context_posthoc
                @ a_posthoc_inv
                @ a_posthoc
                @ a_posthoc_inv
                @ self.checkpoint.a_context_posthoc.T
            )
            solve_b = (
                solve_b
                + self.checkpoint.a_context_posthoc
                @ a_posthoc_inv
                @ self.checkpoint.b_posthoc
            )
        # Solve for linear policy vector
        self.checkpoint.linear_model = np.linalge.solve(solve_a, solve_b)

    @override
    def get_checkpoint(self) -> Any:
        # Docstring copied from @override

        return self.checkpoint

    @override
    def set_checkpoint(self, checkpoint: Any) -> bool:
        # Docstring copied from @override

        # Validation
        if not isinstance(checkpoint, LinearPolicyCheckpoint):
            logger.error("Checkpoint type mismatch in linear policy.")
            return False
        if checkpoint.n_actions != self.checkpoint.n_actions:
            logger.error("Checkpoint n_actions mismatch in linear policy.")
            return False
        if checkpoint.context_dim != self.checkpoint.context_dim:
            logger.error("Checkpoint context_dim mismatch in linear policy.")
            return False
        if checkpoint.posthoc_dim != self.checkpoint.posthoc_dim:
            logger.error("Checkpoint context_dim mismatch in linear policy.")
            return False

        # Load Parameters
        self.checkpoint.a_context = checkpoint.a_context
        self.checkpoint.b_context = checkpoint.b_context
        self.checkpoint.a_posthoc = checkpoint.a_posthoc
        self.checkpoint.b_posthoc = checkpoint.b_posthoc
        self.checkpoint.a_context_posthoc = checkpoint.a_context_posthoc

        if checkpoint.library != self.checkpoint.library:
            logger.warning(
                "Checkpoint library mismatch in linear policy. \
                We can continue, but the policy may be inaccurate..."
            )

        if (
            checkpoint.use_posthoc != self.checkpoint.use_posthoc
            or checkpoint.lambda_l2 != self.checkpoint.lambda_l2
            or checkpoint.lambda_posthoc_damp != self.checkpoint.lambda_posthoc_damp
        ):
            logger.warning(
                "Checkpoint lambdas or posthoc augmentation mismatch in linear policy. \
                Initial re-solve required..."
            )
            self._solve()
        else:
            self.checkpoint.linear_model = checkpoint.linear_model

        return True


RandomPolicy = LinearPolicy
