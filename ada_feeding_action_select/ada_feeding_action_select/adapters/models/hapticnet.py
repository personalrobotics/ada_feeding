#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the HapticNet
Mostly copied from:
https://github.com/personalrobotics/posthoc_learn
"""

# Standard imports
from dataclasses import dataclass

# Third-party imports
from torch import nn
from torch.nn import functional as F


@dataclass
class HapticNetConfig:
    """
    Data Class for configuring HapticNet
    """

    # Layer Config
    n_input_dim: int = 6
    n_input_len: int = 64
    n_output: int = 4
    dropout: float = 0.1


class HapticNet(nn.Module):
    """
    HapticNet, MLP originally for classifying
    food haptic categories
    """
    def __init__(self, config: HapticNetConfig = HapticNetConfig()):
        super().__init__()
        self.config = config

        self.linear = nn.Sequential(
            nn.Linear(self.config.n_input_dim * self.config.n_input_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.n_output),
        )

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, haptic_data):
        """
        Run Network on input data
        """

        output = self.linear(
            haptic_data.view(-1, self.config.n_input_dim * self.config.n_input_len)
        )
        if self.config.dropout > 0 and self.training:
            output = F.dropout(output, training=self.training, p=self.config.dropout)
        output = self.softmax(output)
        return output

    # TODO: Add preprocess from https://github.com/personalrobotics/posthoc_learn/blob/main/src/posthoc_learn/haptic.py
