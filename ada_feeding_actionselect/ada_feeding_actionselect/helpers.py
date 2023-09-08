#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a number of helper functions that are reused throughout
ada_feeding_actionselect.
"""

# Standard imports
import os
import yaml
from typing import List

# Third-party imports
from ament_index_python.packages import get_package_share_directory

# Local imports
from ada_feeding_msgs.msg import AcquisitionSchema

def get_action_library(
    library_path: str, 
    key: str = "actions"
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

    package_share = get_package_share_directory('ada_feeding_actionselect')
    filename = os.path.join(package_share, library_path)

    yaml_file = None
    with open(filename, 'r') as file:
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
        schema.grasp_twist.linear.x = element["grasp_offset"][0]
        schema.grasp_twist.linear.y = element["grasp_offset"][1]
        schema.grasp_twist.linear.z = element["grasp_offset"][2]
        schema.grasp_twist.angular.x = element["grasp_rot"][0]
        schema.grasp_twist.angular.y = element["grasp_rot"][1]
        schema.grasp_twist.angular.z = element["grasp_rot"][2]

        schema.grasp_duration.sec = int(element["grasp_duration"])
        decimal = element["grasp_duration"] - schema.grasp_duration.sec
        sehcma.grasp_duration.nanosec = (decimal * 1000000000) % 1000000000

        schema.grasp_force = element["grasp_force"]
        schema.grasp_torque = element["grasp_torque"]

        # Extraction
        schema.ext_twist.linear.x = element["ext_offset"][0]
        schema.ext_twist.linear.y = element["ext_offset"][1]
        schema.ext_twist.linear.z = element["ext_offset"][2]
        schema.ext_twist.angular.x = element["ext_rot"][0]
        schema.ext_twist.angular.y = element["ext_rot"][1]
        schema.ext_twist.angular.z = element["ext_rot"][2]

        schema.ext_duration.sec = int(element["ext_duration"])
        decimal = element["ext_duration"] - schema.ext_duration.sec
        sehcma.ext_duration.nanosec = (decimal * 1000000000) % 1000000000

        schema.ext_force = element["ext_force"]
        schema.ext_torque = element["ext_torque"]

        library.append(schema)

    return library
