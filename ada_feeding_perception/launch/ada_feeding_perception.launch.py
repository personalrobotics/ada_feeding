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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)


# pylint: disable=too-many-locals
# This is a launch file, so it's okay to have a lot of local variables.
def generate_launch_description():
    """Creates launch description to run all perception nodes"""
    launch_description = LaunchDescription()

    # Get the ada_feeding_perception share directory
    ada_feeding_perception_share_dir = get_package_share_directory(
        "ada_feeding_perception"
    )

    # Whether to combine all perception nodes into one
    combine_perception_nodes_da = DeclareLaunchArgument(
        "combine_perception_nodes",
        default_value="false",
        description="Whether to combine all perception nodes into one",
    )
    combine_perception_nodes = LaunchConfiguration("combine_perception_nodes")
    launch_description.add_action(combine_perception_nodes_da)

    # Declare launch arguments
    use_republisher_da = DeclareLaunchArgument(
        "use_republisher",
        default_value="true",
        description="Whether to use the republisher node",
    )
    use_republisher = LaunchConfiguration("use_republisher")
    launch_description.add_action(use_republisher_da)

    # If we are using the republisher, add the republisher node
    republisher_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "republisher.yaml"
    )
    republisher = Node(
        package="ada_feeding_perception",
        name="republisher",
        executable="republisher",
        parameters=[republisher_config],
        condition=IfCondition(use_republisher),
    )
    launch_description.add_action(republisher)

    # Add the nano_bridge receiver node
    nano_bridge_receiver = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    get_package_share_directory("nano_bridge"),
                    "launch",
                    "receiver.launch.xml",
                ]
            ),
        ),
    )
    launch_description.add_action(nano_bridge_receiver)

    # Remap from the perception nodes to the realsense topics
    prefix = "/local"  # NOTE: must match the topic names in the yaml file!
    realsense_remappings = [
        (
            "~/image",
            PythonExpression(
                expression=["'", prefix, "/camera/color/image_raw/compressed'"]
            ),
        ),
        (
            "~/camera_info",
            PythonExpression(
                expression=["'", prefix, "/camera/aligned_depth_to_color/camera_info'"]
            ),
        ),
    ]
    aligned_depth_remapping = [
        (
            "~/aligned_depth",
            PythonExpression(
                expression=[
                    "'",
                    prefix,
                    "/camera/aligned_depth_to_color/image_raw/compressedDepth'",
                ]
            ),
        ),
    ]

    # Load the segment from point node
    segment_from_point_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "segment_from_point.yaml"
    )
    segment_from_point_params = {}
    segment_from_point_params["segment_from_point_model_dir"] = ParameterValue(
        os.path.join(ada_feeding_perception_share_dir, "model"), value_type=str
    )
    segment_from_point = Node(
        package="ada_feeding_perception",
        name="segment_from_point",
        executable="segment_from_point",
        parameters=[segment_from_point_config, segment_from_point_params],
        remappings=realsense_remappings + aligned_depth_remapping,
        condition=UnlessCondition(combine_perception_nodes),
    )
    launch_description.add_action(segment_from_point)

    # Load the face detection node
    face_detection_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "face_detection.yaml"
    )
    face_detection_params = {}
    face_detection_params["face_detection_model_dir"] = ParameterValue(
        os.path.join(ada_feeding_perception_share_dir, "model"), value_type=str
    )
    # To avoid incorrect depth estimates from the food on the fork, face detection
    # uses the depth image that has been filtered for the octomap, where those
    # points have been set to 0.
    face_detection_remappings = [
        ("~/face_detection", "/face_detection"),
        ("~/face_detection_img/compressed", "/face_detection_img/compressed"),
        ("~/toggle_face_detection", "/toggle_face_detection"),
        (
            "~/aligned_depth_no_fork",
            PythonExpression(
                expression=[
                    "'",
                    prefix,
                    "/camera/aligned_depth_to_color/depth_octomap'",
                ]
            ),
        ),
    ]
    face_detection = Node(
        package="ada_feeding_perception",
        name="face_detection",
        executable="face_detection",
        parameters=[face_detection_config, face_detection_params],
        remappings=realsense_remappings + face_detection_remappings,
        condition=UnlessCondition(combine_perception_nodes),
    )
    launch_description.add_action(face_detection)

    # Load the table detection node
    table_detection_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "table_detection.yaml"
    )
    table_detection_remappings = [
        ("~/table_detection", "/table_detection"),
        ("~/plate_detection_img", "/plate_detection_img"),
        ("~/toggle_table_detection", "/toggle_table_detection"),
    ]
    table_detection = Node(
        package="ada_feeding_perception",
        name="table_detection",
        executable="table_detection",
        parameters=[table_detection_config],
        remappings=realsense_remappings
        + aligned_depth_remapping
        + table_detection_remappings,
        condition=UnlessCondition(combine_perception_nodes),
    )
    launch_description.add_action(table_detection)

    # Load the food-on-fork detection node
    food_on_fork_detection_config = os.path.join(
        ada_feeding_perception_share_dir, "config", "food_on_fork_detection.yaml"
    )
    food_on_fork_detection_params = {}
    food_on_fork_detection_params["food_on_fork_detection_model_dir"] = ParameterValue(
        os.path.join(ada_feeding_perception_share_dir, "model"), value_type=str
    )
    food_on_fork_detection_remappings = [
        ("~/food_on_fork_detection", "/food_on_fork_detection"),
        ("~/food_on_fork_detection_img", "/food_on_fork_detection_img"),
        ("~/toggle_food_on_fork_detection", "/toggle_food_on_fork_detection"),
    ]
    food_on_fork_detection = Node(
        package="ada_feeding_perception",
        name="food_on_fork_detection",
        executable="food_on_fork_detection",
        parameters=[food_on_fork_detection_config, food_on_fork_detection_params],
        remappings=realsense_remappings
        + aligned_depth_remapping
        + food_on_fork_detection_remappings,
        condition=UnlessCondition(combine_perception_nodes),
    )
    launch_description.add_action(food_on_fork_detection)

    # Load the combined perception node
    ada_feeding_perception = Node(
        package="ada_feeding_perception",
        name="ada_feeding_perception",
        executable="ada_feeding_perception_node",
        parameters=[
            segment_from_point_config,
            segment_from_point_params,
            face_detection_config,
            face_detection_params,
            table_detection_config,
            food_on_fork_detection_config,
            food_on_fork_detection_params,
        ],
        remappings=realsense_remappings
        + aligned_depth_remapping
        + face_detection_remappings
        + table_detection_remappings
        + food_on_fork_detection_remappings,
        condition=IfCondition(combine_perception_nodes),
    )
    launch_description.add_action(ada_feeding_perception)

    return launch_description
