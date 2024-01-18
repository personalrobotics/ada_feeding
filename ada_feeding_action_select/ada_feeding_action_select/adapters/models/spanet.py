#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SPANet featurizer.
Mostly copied from:
https://github.com/personalrobotics/bite_selection_package
"""

# Standard imports
from dataclasses import dataclass

# Third-party imports
import numpy.typing as npt
from PIL import Image as PILImage
import torch
from torch import nn
from torchvision import transforms


@dataclass
class SPANetConfig:
    """
    Data Class for configuring SPANet
    """

    # Which data to use
    use_rgb: bool = True
    use_depth: bool = False
    use_wall: bool = True

    # Layer Config
    image_size: int = 144
    n_linear_size: int = 2048
    n_features: int = 2048
    final_vector_size: int = 10


class SPANet(nn.Module):
    """
    SPANet Model
    """

    # pylint: disable=too-many-instance-attributes
    # Lots of layer definitions in a NN

    def __init__(self, config: SPANetConfig = SPANetConfig()):
        """
        Init Network
        """

        super().__init__()
        self.config = config

        self.conv_init_rgb = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # 144
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_init_depth = nn.Sequential(
            nn.Conv2d(1, 16, 11, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # 144
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_merge = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_layers_top = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 72
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # 36
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),  # 18
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_layers_bot = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),  # 9
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        n_features = config.n_linear_size
        n_features_final = 2048
        if config.n_features is not None:
            n_features_final = config.n_features

        if config.use_wall:
            n_flattened = 9 * 9 * 256 + 3
        else:
            n_flattened = 9 * 9 * 256

        self.linear_layers = nn.Sequential(
            nn.Linear(n_flattened, n_features),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features_final),
            nn.BatchNorm1d(n_features_final),
            nn.ReLU(),
        )

        self.final = nn.Linear(n_features_final, config.final_vector_size)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_image(self, image: npt.NDArray, use_cuda: bool = True) -> torch.Tensor:
        """
        Crop/pad image and prepare for sending to SPANet
        """
        img_org = PILImage.fromarray(image.copy())
        ratio = float(self.config.image_size / max(img_org.size))
        new_size = tuple(int(x * ratio) for x in img_org.size)
        pads = [
            (self.config.image_size - new_size[0]) // 2,
            (self.config.image_size - new_size[1]) // 2,
        ]
        img_org = img_org.resize(new_size, PILImage.ANTIALIAS)
        img = PILImage.new("RGB", (self.config.image_size, self.config.image_size))
        img.paste(img_org, pads)
        return torch.stack(
            [self.transform(img).cuda() if use_cuda else self.transform(img).cpu()]
        )

    def forward(self, rgb, depth, loc_type=None):
        """
        Run Network on input image
        """

        out_rgb, out_depth = None, None
        if self.config.use_rgb:
            out_rgb = self.conv_init_rgb(rgb)
        if self.config.use_depth:
            out_depth = self.conv_init_depth(depth)

        if self.config.use_rgb and self.config.use_depth:
            merged = torch.cat((out_rgb, out_depth), 1)
            out = self.conv_merge(merged)
        else:
            out = out_rgb if out_rgb is not None else out_depth

        out = self.conv_layers_top(out)
        for _ in range(3):
            out = self.conv_layers_bot(out) + out

        out = out.view(-1, 9 * 9 * 256)

        # Add Wall Detector
        if loc_type is None:
            loc_type = torch.tensor([[1.0, 0.0, 0.0]]).repeat(
                out.size()[0], 1
            )  # Isolated = default
            if out.is_cuda:
                loc_type = loc_type.cuda()

        if self.config.use_wall:
            out = torch.cat((out, loc_type), dim=1)

        out = self.linear_layers(out)
        features = out.clone().detach()

        out = self.final(out)

        out = out.sigmoid()
        return out, features
