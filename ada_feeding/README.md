# ada_feeding

The `ada_feeding` package contains the overarching code to run the robot-assisted feeding system. This includes code to plan and execute motions on the robot, code to communicate with the app, and code that reasons over the results of perception. The only code this does not contain is the perception code.

## Getting Started

This code has been developed and tested with the Kinova JACO Gen2 Arm, on computers running Ubuntu 22.04. To get started:
- Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- Install Python dependencies: `python3 -m pip install pyyaml py_trees pymongo tornado trimesh`
- Install the code to command the real robot ([instructions here](https://github.com/personalrobotics/ada_ros2/blob/main/README.md))
- Git clone the [PRL fork of pymoveit (branch: `amaln/allowed_collision_matrix`)](https://github.com/personalrobotics/pymoveit2) and the [PRL fork of py_trees_ros (branch: `amaln/service_client`)](https://github.com/personalrobotics/py_trees_ros/tree/amaln/service_client) into your ROS2 workspace's `src` folder.
- Install additional dependencies: `sudo apt install ros-humble-py-trees-ros-interfaces`.
- Install the web app into your workspace ([instructions here](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp)).

## Usage
1. Build your workspace: `colcon build`
2. Source your workspace: `source install/setup.bash`
3. Launch the force-torque sensor:
    1. Dummy node: `ros2 run ada_feeding dummy_ft_sensor.py`
    2. Real node: Follow the instructions in the [`forque_sensor_hardware` README](https://github.com/personalrobotics/forque_sensor_hardware/blob/main/README.md). Note that this is in the _main_ branch, which may not be the default branch.
4. Launch MoveIt2:
    1. Sim (RVIZ): `ros2 launch ada_moveit demo.launch.py sim:=mock`
    2. Real Robot: `ros2 launch ada_moveit demo.launch.py`
5. Launch the RealSense & Perception:
    1. Dummy nodes: `ros2 launch feeding_web_app_ros2_test feeding_web_app_dummy_nodes_launch.xml run_motion:=false run_web_bridge:=false`
    2. Real nodes: Follow the instructions in the [`ada_feeding_perception` README](https://github.com/personalrobotics/ada_feeding/blob/ros2-devel/ada_feeding_perception/README.md#usage)
6. Run the action servers: `ros2 launch ada_feeding ada_feeding_launch.xml`
7. Test it:
    1. Test the individual actions with the command line interface:
        1. `ros2 action send_goal /MoveAbovePlate ada_feeding_msgs/action/MoveTo "{}" --feedback`
        2. `ros2 action send_goal /AcquireFood ada_feeding_msgs/action/AcquireFood "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, detected_food: {roi: {x_offset: 0, y_offset: 0, height: 0, width: 0, do_rectify: false}, mask: {header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, format: '', data: []}, item_id: '', confidence: 0.0}}" --feedback`
        3. `ros2 action send_goal /MoveToRestingPosition ada_feeding_msgs/action/MoveTo "{}" --feedback`
        4. `ros2 action send_goal /MoveToMouth ada_feeding_msgs/action/MoveTo "{}" --feedback`
        5. `ros2 action send_goal /MoveToStowLocation ada_feeding_msgs/action/MoveTo "{}" --feedback`
    2. Test the individual actions with the web app:
        1. Launch the web app ([instructions here](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp))
        2. Go through the web app, ensure the expected actions happen on the robot.
    3. Test the watchdog in isolation:
        1. Echo the watchdog topic: `ros2 topic echo /ada_watchdog`
        2. Induce errors in the force-torque sensor and verify the watchdog reacts appropiately. Errors could include:
            - Terminating the node.
            - Disconnecting the physical force-torque sensor from power.
            - Inducing corruption by causing the dummy note to output zero-variance values: `ros2 param set /dummy_ft_sensor std  '[0.1, 0.1, 0.0, 0.1, 0.1, 0.1]'`
    4. Test the watchdog with the action servers:
        1. Start an action (see above), induce errors in the force-torque sensor (see above), and ensure the action gets aborted.
        2. While there are errors in the force-torque sensor, ensure no new goals get accepted.
        3. Terminate the watchdog node and ensure in-progress actions get aborted and incoming goals get rejected.
        4. Launch the action servers without the watchdog node running and ensure it rejects all goals.

## Writing Behavior Trees That Can Be Wrapped Into Action Servers

### Subclassing `ActionServerBT`

In order to wrap a behavior tree into an action server, you have to subclass `ActionServerBT`. Your subclass must implement the following functions:
- `create_tree`: Create the tree. This is called **once**, sometime before the first call to this action server begins executing.
- `send_goal`: Take a goal from the action server and send it to the tree. This is called **once per goal** that is sent to this action server.
- `get_feedback`: Return the feedback type that the ROS2 action is expecting, by examining the tree's current state. This is called **once per action server iteration** (which is the same as being called once per `tree.tick()`).
- `get_result`: Return the result type that the ROS2 action is expecting, by examining the tree's current state. This is called **once per goal**, after the tree's status has switched away from `RUNNING`.

Note the following design paradigms when writing this subclass of `ActionServerBT`:
- None of the above functions should be blocking.
- All communication between this file and the behavior tree should take place through the blackboard.
- Only this file should need to import the ROS action type for the action server. The specific behaviors in the tree should not need access to that information.
- ROS action preemption requests are implemented using the `stop()` pathway of the behavior tree, which internally calls `terminate()` on each behavior. Hence, `terminate()` must properly implement everything that needs to be done when the ROS action preempts. (It should already be implemented this way for good behavior tree design.)

### Subclassing `MoveToTree`

All robot motion actions publish the same feedback and result (see [`ada_feeding_msgs`](https://github.com/personalrobotics/ada_feeding/tree/amaln/goal_constraints/ada_feeding_msgs/action)). Hence, the shared logic of publishing that feedback and result is implemented in the `MoveToTree`. Any tree that publishes that feedback and result should subclass `MoveToTree` and  implement the abstract function `create_move_to_tree`.

### Creating Robot Motions with Arbitrary Goal and Path Constraints
Robot motion is implemented as a behavior, `MoveTo`. By itself, the behavior will do nothing, since no goal or path constraints will be set. Goal and path constraints are implemented as decorators on top of the `MoveTo` behavior, that can be chained together in order to have multiple constraints. See the already implemented trees (e.g., `MoveToConfiguration`, `MoveToConfigurationWithPosePathConstraints`) for examples.

### Updating the Config File

Then, to execute the behavior tree in `create_action_servers.py`, you have to modify the ROS param config file (e.g., `ada_feeding_action_servers.yaml`). The config file has the following structure:

- `server_names` (string array, required): A list of the names of the action servers to create.
- For each server name:
    - `action_type` (string, required): the ROS action type associated with this action, e.g., `ada_feeding_msgs.action.MoveTo`
    - `tree_class` (string, required): the subclass of `ActionServerBT` that creates the tree and provides functions to interface with the action server, e.g., `ada_feeding.trees.MoveAbovePlate`
    - `tree_kws` (string array, optional, default `[]`): a list with the keywords for custom arguments to pass to the initialization function of the `tree_class`.
    - `tree_kwargs` (dict, optional, default `{}`): a dictionary of keyword arguments to pass to the initialization function of the `tree_class`.
    - `tick_rate` (int, optional, default `30`): the frequency (Hz) at which the action server should tick the tree.

Once you've made the above changes, you should be good to go. Rebuild your workspace and run the launchfile!

## Troubleshooting

- If you get `Error: the following exception was never retrieved: cannot use destroyable because destruction was requested` that likely means you are running an old version of `rclpy`. Upgrading `rclpy` should address the issue.
