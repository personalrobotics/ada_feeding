from glob import glob
import os
from setuptools import find_packages, setup

package_name = "nano_bridge"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        # Include all config files.
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Amal Nanavati",
    maintainer_email="amaln@uw.edu",
    description=(
        "This package contains a bridge that allows ROS2 to work well over WiFi with "
        "our Jetson Nano. This was developed in response to an issue where, when we "
        "subscribe to too many camera topics on the nano (e.g., one or more each for a "
        "compressed RGB and compressed depth image), one of those topics will stop publishing "
        "entirely for minutes on end. The closest Github Issues are linked from: "
        "https://github.com/personalrobotics/ada_feeding/issues/73"
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "nano_bridge_receiver = nano_bridge.receiver:main",
            "nano_bridge_sender = nano_bridge.sender:main",
        ],
    },
)
