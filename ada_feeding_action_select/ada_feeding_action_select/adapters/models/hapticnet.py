#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the HapticNet
Mostly copied from:
https://github.com/personalrobotics/posthoc_learn
"""

# Standard imports
from dataclasses import dataclass
from typing import List, Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms


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


# Crop Data
@dataclass
class CropConfig:
    """
    Data Class for config of crop()
    """

    force_thresh: float = 0.1  # % of max force for contact
    force_inc: float = 0.01  # Amount to increase thresh each crop loop
    min_len: int = 20  # Minimum length to crop
    recurse_pad: int = 10  # how far after noise to check each loop
    lookahead: List[int] = [
        15,
        30,
    ]  # Indices to lookahead to check for drop below thresh


def crop(
    data: npt.NDArray,
    z_dim: int = 3,
    config: CropConfig = CropConfig(),
    # Recurse Vars
    start_idx: int = 0,
):
    """
    Pre-truncates data to contact point:
    a sharp increase in Z force
    Based on:
    https://github.com/personalrobotics/posthoc_learn/blob/main/src/posthoc_learn/haptic.py
    In turn based on Bhattacharjee et al., 2019 HapticNet preprocessing

    Parameters
    ----------
    data: n_dims x n_time np array
    z_dim: Z dimension index
    config: other magic numbers, see above
    """

    # pylint: disable=too-many-branches, too-many-locals
    # Not the *cleanest* impl, but wanted to copy the logic
    # from github with minimal changes
    z_force = np.abs(np.copy(data[z_dim, :]))

    max_force = max(z_force)
    min_force = min(z_force)
    force_thr = config.force_thresh * max_force
    crop_flag = False
    idx_init = idx_end = -1

    while abs(idx_end - idx_init) <= config.min_len:
        for idx, z in enumerate(z_force, start_idx):
            prev_crop_flag = crop_flag
            if float(z) > force_thr:
                # Exceed thresh, start crop
                crop_flag = True
                if not prev_crop_flag:
                    idx_init = idx - 1
            else:
                lookahead_dip = True
                # Lookahead to avoid noisy dip
                for lookahead in config.lookahead:
                    if (idx + lookahead) < len(z_force) and float(
                        z_force[idx + lookahead]
                    ) >= force_thr:
                        lookahead_dip = False
                        break
                if lookahead_dip:
                    crop_flag = False
                    if prev_crop_flag:
                        # True Dip, End Cropping
                        idx_end = idx
            # Cropping Happened, break
            if prev_crop_flag != crop_flag and crop_flag == 0:
                break
        # No cropping happened
        if idx_init < 0:
            break
        # Crop was too short, maybe noise bump?
        # Try again with higher force threshold
        force_thr = force_thr + config.force_inc

    if idx_init < 0:
        print("cropping failed")
        return None

    z_force_truncated = z_force[idx_init:idx_end]
    if (
        max(z_force_truncated) - min(z_force_truncated) < (max_force - min_force) * 0.5
    ):  # it was just noise after all, try next peak
        return crop(data, z_dim, config, start_idx=idx_end + config.recurse_pad)

    # Return start crop to end of data
    idx_end = int(len(z_force))
    return (idx_init, idx_end)


def resize(data: torch.Tensor, out_dim: int) -> torch.Tensor:
    """
    Resize input tensor to output dimension
    """
    n_features = data[0].shape[0]
    processed_data = []

    data_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((n_features, out_dim))]
    )

    for x in data:
        img = x.view(x.size(0), x.size(2)).cpu().data.numpy()[:, :, None]
        out = data_transform(img)
        out = Variable(torch.from_numpy(np.asarray(out)).float().cuda())
        if out.size(0) == out_dim:
            out = out.permute(1, 0)
        processed_data += [out]

    return processed_data


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

    def preprocess(self, data: npt.NDArray, z_dim: int = 3) -> Optional[torch.Tensor]:
        """
        Take haptic data and crop it to the contact point.
        Input data can be transposed if # of samples != n_input_dim+1

        Parameters
        ----------
        data: numpy array of data ([n_input_dim + 1] X # of samples or transpose)
        z_dim: dimension of the z-force used for cropping. Default 3
                (time, fx, fy, fz, ...)

        Return
        ------

        The data ready to be passed into forward(), or None if crop fails

        """
        assert len(data.shape) == 2
        crop_data = np.copy(data)
        if crop_data.shape[0] != (self.config.n_input_dim + 1):
            crop_data = np.transpose(crop_data)
        if crop_data.shape[0] != (self.config.n_input_dim + 1):
            # Incorrect dimensionality
            return None

        crop_range = crop(crop_data, z_dim)
        if crop_range is None:
            return None
        return resize(
            torch.from_numpy(crop_data[1:, crop_range[0] : crop_range[1]]).float(),
            self.config.n_input_len,
        )
