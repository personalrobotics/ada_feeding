"""
This package contains code that adapts incoming sensor info to context
and posthoc context.
"""

# Generic Adapter Classes
from .base_adapters import ContextAdapter, PosthocAdapter

# Just returns [0]
from .base_adapters import NoContext

# TODO: SegmentAnything Context

# SPANet Context
from .spanet_adapter import SPANetContext

# Color Context
from .color_adapter import ColorContext

# HapticNet Posthoc
from .hapticnet_adapter import HapticNetPosthoc
