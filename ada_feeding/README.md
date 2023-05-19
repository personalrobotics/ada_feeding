# ada_feeding

The `ada_feeding` package contains the overarching code to run the robot-assisted feeding system. This includes code to plan and execute motions on the robot, code to communicate with the app, and code that reasons over the results of perception. The only code this does not contain is the perception code.

## Getting Started

This code has been developed and tested with the Kinova JACO Gen2 Arm, on computers running Ubuntu 22.04. To get started:
- Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- Install Python dependencies: `python3 -m pip install pyyaml`

## Usage
- Run the action servers: `ros2 launch ada_feeding ada_feeding_launch.xml`
- Test the action servers:
    - `ros2 action send_goal /MoveAbovePlate ada_feeding_msgs/action/MoveTo {}\`
    - `ros2 action send_goal /AcquireFood ada_feeding_msgs/action/AcquireFood "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, detected_food: {roi: {x_offset: 0, y_offset: 0, height: 0, width: 0, do_rectify: false}, mask: {header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, format: '', data: []}, item_id: '', confidence: 0.0}}"`