"""
This package contains code that adapts incoming sensor info to context
and posthoc context.
"""

# Generic Adapter Classes
from .base_adapters import ContextAdapter, PosthocAdapter

# Just returns [0]
from .base_adapters import NoContext

# TODO: Posthoc Passthrough

# SPANet Context
from .spanet_adapter import SPANetAdapter

# HapticNet Posthoc
from .hapticnet_adapter import HapticNetAdapter
