from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ada_feeding_actionselect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all config files.
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
        # Include any previous data / model checkpoints
        (
            os.path.join("share", package_name, "data"),
            glob(os.path.join("data", "*.npz")),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ethan K. Gordon',
    maintainer_email='ekgordon@cs.washington.edu',
    description='Framework for selecting motion primitives (e.g. food acquisition actions) from food visual and haptic information.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "policy_service = ada_feeding_actionselect.policy_service:main",
        ],
    },
)
