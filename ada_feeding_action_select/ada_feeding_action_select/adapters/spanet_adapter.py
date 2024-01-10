#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SPANet context adapter.

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
from ada_feeding_msgs.msg import Mask
from ada_feeding_perception.helpers import ros_msg_to_cv2_image
from .models import SPANetConfig, SPANet
from .base_adapters import ContextAdapter


class SPANetContext(ContextAdapter):
    """
    An adapter to run images through SPANet
    and extract features.
    """

    def __init__(
        self,
        checkpoint: str,
        n_features: int = 2048,
        gpu_index: int = 0,
    ) -> None:
        """
        Load Checkpoint and Set Config Parameters

        Parameters
        ----------
        checkpoint: PTH file relative to share directory / data
        n_features: size of the SPANet feature vectory (determined by checkpoint)
        gpu_index: which gpu to use for CUDA
        """

        # Init CUDA
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("Init SPANet with CUDA")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        # Init SPANet
        self.config = SPANetConfig(n_features=n_features)
        self.spanet = SPANet(self.config)

        # Load Checkpoint
        ckpt_file = os.path.join(
            get_package_share_directory("ada_feeding_action_select"), "data", checkpoint
        )
        ckpt = torch.load(ckpt_file)
        self.spanet.load_state_dict(ckpt["net"])
        self.spanet.eval()
        if self.use_cuda:
            self.spanet = self.spanet.cuda()

    @property
    @override
    def dim(self) -> int:
        # Docstring copied from @override
        # Add Bias to features
        return self.config.n_features + 1

    @override
    def get_context(self, mask: Mask) -> npt.NDArray:
        # Docstring copied from @override

        # Prepare image(s)
        img = None
        if self.config.use_rgb:
            img = self.spanet.prepare_image(
                ros_msg_to_cv2_image(mask.rgb_image), self.use_cuda
            )
        depth = None
        if self.config.use_depth:
            depth = self.spanet.prepare_image(
                ros_msg_to_cv2_image(mask.depth_image), self.use_cuda
            )

        # Get SPANet Features
        _, features = self.spanet(img, depth)

        # Flatten and add Bias
        features_flat = features.cpu().detach().numpy().flatten().tolist()
        features_flat.insert(0, 1.0)
        ret = np.array(features_flat)

        assert ret.size == self.dim
        return ret
