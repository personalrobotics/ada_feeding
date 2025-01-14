# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This visitor logs the name and status of the behavior it is on.
"""

# Standard imports

# Third-party imports
from py_trees.behaviour import Behaviour
from py_trees.visitors import VisitorBase

# Local imports


class DebugVisitor(VisitorBase):
    """
    This visitor logs the name and status of each behavior it visits.
    """

    def run(self, behaviour: Behaviour) -> None:
        """
        Log behaviour information on the debug channel.

        Args:
            behaviour: behaviour being visited.
        """
        behaviour.logger.debug(f"{behaviour.name} [{behaviour.status}]")
