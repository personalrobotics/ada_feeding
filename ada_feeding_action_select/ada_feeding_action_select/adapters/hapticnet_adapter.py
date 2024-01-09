#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the HapticNet context adapter.

"""

# Standard imports
import os

# Third-party imports
from ament_index_python.packages import get_package_share_directory
import numpy as np
import numpy.typing as npt
from overrides import override
import torch

# Local imports
from .models import HapticNetConfig, HapticNet
from .base_adapters import PosthocAdapter


class HapticNetAdapter(PosthocAdapter):
    """
    An adapter to run force/torque data through HapticNet
    and extract features.
    """

    def __init__(
        self,
        checkpoint: str,
        n_features: int = 4,
        gpu_index: int = 0,
    ) -> None:
        """
        Load Checkpoint and Set Config Parameters

        Parameters
        ----------
        checkpoint: PTH file relative to share directory / data
        n_features: size of the HapticNet feature vectory (determined by checkpoint)
        gpu_index: which gpu to use for CUDA
        """

        # Init CUDA
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("Init HapticNet with CUDA")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        # Init HapticNet
        self.config = HapticNetConfig(n_output=n_features)
        self.hapticnet = HapticNet(self.config)

        # Load Checkpoint
        ckpt_file = os.path.join(
            get_package_share_directory("ada_feeding_action_select"), "data", checkpoint
        )
        ckpt = torch.load(ckpt_file)
        self.hapticnet.load_state_dict(ckpt["state_dict"])
        self.hapticnet.eval()
        if self.use_cuda:
            self.hapticnet = self.hapticnet.cuda()

    @property
    @override
    def dim(self) -> int:
        # Docstring copied from @override
        # No Bias: Haptic bias apparently made it much, much worse
        return self.config.n_output

    @override
    def get_posthoc(self, data: npt.NDArray) -> npt.NDArray:
        # Docstring copied from @override

        ret = np.array([])

        # Run data through hapticnet
        input_data = self.hapticnet.preprocess(data)
        if input_data is None:
            return ret
        if self.use_cuda:
            input_data = input_data.cuda()

        # Get HapticNet Features
        features = self.hapticnet(input_data)

        # Flatten, no bias
        ret = features.cpu().detach().numpy().flatten()

        assert ret.size == self.dim
        return ret
