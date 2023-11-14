#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch file for ada_feeding_perception
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


# pylint: disable=too-many-locals
# This is a launch file, so it's okay to have a lot of local variables.
def generate_launch_description():
    """Creates launch description to run all perception nodes"""
    launch_description = LaunchDescription()

    # Get the ada_feeding_perception share directory
    ada_feeding_perception_share_dir = get_package_share_directory(
        "ada_feeding_perception"
    )

    # Declare launch arguments
    use_republisher_da = DeclareLaunchArgument(
        "use_republisher",
        default_value="true",
        description="Whether to use the republisher node",
    )
    use_republisher = LaunchConfiguration("use_republisher")
    launch_description.add_action(use_republisher_da)
    republished_namespace_da = DeclareLaunchArgument(
        "republished_namespace",
        default_value="/local",
        description="The namespace to republish topics under.",
    )
    republished_namespace = LaunchConfiguration("republished_namespace")
    launch_description.add_action(republished_namespace_da)

    # If we are using the republisher, add the republisher node
    republisher_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "republisher.yaml"
    )
    republisher_params = {}
    republisher_params["republished_namespace"] = ParameterValue(
        republished_namespace, value_type=str
    )
    republisher = Node(
        package="ada_feeding_perception",
        name="republisher",
        executable="republisher",
        parameters=[republisher_config, republisher_params],
        condition=IfCondition(use_republisher),
    )
    launch_description.add_action(republisher)

    # Remap from the perception nodes to the realsense topics
    prefix = PythonExpression(
        expression=[
            "'",
            republished_namespace,
            "' if '",
            use_republisher,
            "'=='true' else ''",
        ]
    )
    realsense_remappings = [
        (
            "~/image",
            PythonExpression(
                expression=["'", prefix, "/camera/color/image_raw/compressed'"]
            ),
        ),
        (
            "~/camera_info",
            PythonExpression(expression=["'", prefix, "/camera/color/camera_info'"]),
        ),
        (
            "~/aligned_depth",
            PythonExpression(
                expression=["'", prefix, "/local/camera/aligned_depth_to_color/depth_octomap'"]
            ),
        ),
    ]

    # Load the segment from point node
    segment_from_point_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "segment_from_point.yaml"
    )
    segment_from_point_params = {}
    segment_from_point_params["model_dir"] = ParameterValue(
        os.path.join(ada_feeding_perception_share_dir, "model"), value_type=str
    )
    segment_from_point = Node(
        package="ada_feeding_perception",
        name="segment_from_point",
        executable="segment_from_point",
        parameters=[segment_from_point_config, segment_from_point_params],
        remappings=realsense_remappings,
    )
    launch_description.add_action(segment_from_point)

    # Load the face detection node
    face_detection_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "face_detection.yaml"
    )
    face_detection_params = {}
    face_detection_params["model_dir"] = ParameterValue(
        os.path.join(ada_feeding_perception_share_dir, "model"), value_type=str
    )
    face_detection_remappings = [
        ("~/face_detection", "/face_detection"),
        ("~/face_detection_img", "/face_detection_img"),
        ("~/toggle_face_detection", "/toggle_face_detection"),
    ]
    face_detection = Node(
        package="ada_feeding_perception",
        name="face_detection",
        executable="face_detection",
        parameters=[face_detection_config, face_detection_params],
        remappings=realsense_remappings + face_detection_remappings,
    )
    launch_description.add_action(face_detection)

    return launch_description
