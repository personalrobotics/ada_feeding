"""
This module contains policies for selecting actions
based on context (visual and posthoc).
"""

# Generic Policy
from .base_policies import Policy

# Constant Policy
from .base_policies import ConstantPolicy

# Linear Policies
from .linear_policies import RandomLinearPolicy

# Color Policy
from .color_policy import ColorPolicy
