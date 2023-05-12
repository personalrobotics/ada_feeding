from glob import glob
import os
from setuptools import setup

package_name = "feeding_web_app_ros2_test"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Add the video used by the dummy RealSense publisher
        (
            "share/data",
            ["data/2022_11_01_ada_picks_up_carrots_camera_compressed_ft_tf.mp4"],
        ),
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Amal Nanavati",
    maintainer_email="amaln@cs.washington.edu",
    description="A minimal ROS publisher, subscriber, service, and action to use to test the web app.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Scripts for the main app
            "DummyRealSense = feeding_web_app_ros2_test.DummyRealSense:main",
            "MoveAbovePlate = feeding_web_app_ros2_test.MoveAbovePlate:main",
            # Scripts for the "TestROS" component
            "listener = feeding_web_app_ros2_test.subscriber:main",
            "reverse_string = feeding_web_app_ros2_test.reverse_string_service:main",
            "sort_by_character_frequency = feeding_web_app_ros2_test.sort_by_character_frequency_action:main",
            "talker = feeding_web_app_ros2_test.publisher:main",
        ],
    },
)
