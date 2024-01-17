# ada_feeding

This repository contains all the code to run ADA's robot-asssited feeding system. Specifically, `ada_feeding_msgs` includes custom message definitions, `ada_feeding_perception` contains all the perception code, and `ada_feeding` contains all the other code (including the overall launchfile to run the robot-assisted feeding system).

See the README in each individual package for more information.

## Launching Code

A convenience script is included in the top-level directory to load all the `screen` sessions with the appropriate code.

To launch the real robot code, from the top-level of your colcon workspace, run `python3 src/ada_feeding/start.py`.

The script has various options: to see them, run `python3 src/ada_feeding/start.py -h`