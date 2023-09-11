#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a number of helper functions that are reused throughout
ada_feeding_action_select.
"""

# Standard imports
import os
from typing import List
import yaml

# Third-party imports
from ament_index_python.packages import get_package_share_directory

# Local imports
from ada_feeding_msgs.msg import AcquisitionSchema


def get_action_library(
    library_path: str, key: str = "actions"
) -> List[AcquisitionSchema]:
    """
    Loads library for actions from a YAML file.

    Parameters
    ----------
    library_path: location of yaml relative to package share
    key: Must be an array of Acquisition Actions

    Returns
    -------
    Array of AcquisitionSchema message objects
    """

    package_share = get_package_share_directory("ada_feeding_action_select")
    filename = os.path.join(package_share, library_path)

    yaml_file = None
    with open(filename, "r", encoding='utf-8') as file:
        yaml_file = yaml.safe_load(file)

    library = []
    for element in yaml_file[key]:
        schema = AcquisitionSchema()

        # Approach
        schema.pre_transform.position.x = element["pre_pos"][0]
        schema.pre_transform.position.y = element["pre_pos"][1]
        schema.pre_transform.position.z = element["pre_pos"][2]

        schema.pre_transform.orientation.w = element["pre_quat"][0]
        schema.pre_transform.orientation.x = element["pre_quat"][1]
        schema.pre_transform.orientation.y = element["pre_quat"][2]
        schema.pre_transform.orientation.z = element["pre_quat"][3]

        schema.pre_offset.x = element["pre_offset"][0]
        schema.pre_offset.y = element["pre_offset"][1]
        schema.pre_offset.z = element["pre_offset"][2]

        schema.pre_force = element["pre_force"]
        schema.pre_torque = element["pre_torque"]

        if "pre_rot_hint" in element.keys():
            schema.pre_rot_hint.x = element["pre_rot_hint"][0]
            schema.pre_rot_hint.y = element["pre_rot_hint"][1]
            schema.pre_rot_hint.z = element["pre_rot_hint"][2]

        # Grasp
        schema.grasp_linear.x = element["grasp_linear"][0]
        schema.grasp_linear.y = element["grasp_linear"][1]
        schema.grasp_linear.z = element["grasp_linear"][2]
        schema.grasp_angular.x = element["grasp_angular"][0]
        schema.grasp_angular.y = element["grasp_angular"][1]
        schema.grasp_angular.z = element["grasp_angular"][2]

        schema.grasp_duration.sec = int(element["grasp_duration"])
        decimal = element["grasp_duration"] - schema.grasp_duration.sec
        schema.grasp_duration.nanosec = int((decimal * 10**9) % 10**9)

        schema.grasp_force = element["grasp_force"]
        schema.grasp_torque = element["grasp_torque"]

        # Extraction
        schema.ext_linear.x = element["ext_linear"][0]
        schema.ext_linear.y = element["ext_linear"][1]
        schema.ext_linear.z = element["ext_linear"][2]
        schema.ext_angular.x = element["ext_angular"][0]
        schema.ext_angular.y = element["ext_angular"][1]
        schema.ext_angular.z = element["ext_angular"][2]

        schema.ext_duration.sec = int(element["ext_duration"])
        decimal = element["ext_duration"] - schema.ext_duration.sec
        schema.ext_duration.nanosec = int((decimal * 1000000000) % 1000000000)

        schema.ext_force = element["ext_force"]
        schema.ext_torque = element["ext_torque"]

        library.append(schema)

    return library
