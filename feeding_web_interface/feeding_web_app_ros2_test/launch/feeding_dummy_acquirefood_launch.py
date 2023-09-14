#!/usr/bin/env python3
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare arguments
    ld.add_action(
        DeclareLaunchArgument(
            "request",
            default_value="above_plate_1_carrot_request.pkl",
            description=".pkl file in data with request object",
        )
    )

    # Initialize Arguments
    request = LaunchConfiguration("request")

    # Add dummy realsense node
    package_prefix = FindPackageShare("feeding_web_app_ros2_test")
    ld.add_action(
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                [package_prefix, "/launch/feeding_web_app_dummy_nodes_launch.xml"]
            ),
            launch_arguments={
                "run_web_bridge": "false",
                "run_food_detection": "false",
                "run_face_detection": "false",
                "run_motion": "false",
                "run_real_sense": "true",
            }.items(),
        )
    )

    # Add Request
    request_path = {
        "request_path": ParameterValue(
            PathJoinSubstitution([package_prefix, "..", "data", request]),
            value_type=str,
        )
    }

    # Run Request Node
    ld.add_action(
        Node(
            package="feeding_web_app_ros2_test",
            executable="AcquireFoodClient",
            name="AcquireFoodClient",
            parameters=[
                request_path,
            ],
            on_exit=Shutdown(),
        )
    )

    return ld
