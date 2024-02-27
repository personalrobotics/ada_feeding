from glob import glob
import os
from setuptools import setup

package_name = "ada_feeding_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Add an empty directory for the models
        (
            os.path.join("share", package_name, "model"),
            [],
        ),
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        # Include all model files.
        (
            os.path.join("share", package_name, "model"),
            glob(os.path.join("model", "*")),
        ),
        # Include all config files.
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
        # Include the test images
        (
            os.path.join("share", package_name, "test_img"),
            glob(os.path.join("test", "food_img", "*.jpg")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bernie Zhu, Amal Nanavati, Taylor Kessler Faulkner",
    maintainer_email="{haozhu, amaln, taylorkf}@cs.washington.edu",
    description="This package contains all the perception code (face and food perception) for the robot feeding system.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "food_on_fork_detection = ada_feeding_perception.food_on_fork_detection:main",
            "republisher = ada_feeding_perception.republisher:main",
            "segment_from_point = ada_feeding_perception.segment_from_point:main",
            "test_segment_from_point = ada_feeding_perception.test_segment_from_point:main",
            "face_detection = ada_feeding_perception.face_detection:main",
            "test_realsense = ada_feeding_perception.test_realsense:main",
        ],
    },
)
