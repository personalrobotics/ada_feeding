#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SPANet featurizer.
Mostly copied from:
https://github.com/personalrobotics/bite_selectin_package
"""

# Standard imports
from dataclasses import dataclass

# Third-party imports

# Local imports


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
