# ada_feeding

The `ada_feeding` package contains the overarching code to run the robot-assisted feeding system. This includes code to plan and execute motions on the robot, code to communicate with the app, and code that reasons over the results of perception. The only code this does not contain is the perception code.

## Getting Started

This code has been developed and tested with the Kinova JACO Gen2 Arm, on computers running Ubuntu 22.04. To get started:
- Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
- Install Python dependencies: `python3 -m pip install pyyaml py_trees`
- Install the web app into your workspace ([instructions here](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp)).

## Usage
1. Build your workspace: `colcon build`
2. Source your workspace: `source install/setup.bash`
3. Run the action servers: `ros2 launch ada_feeding ada_feeding_launch.xml`
4. Test it:
    1. Individual actions:
        1. `ros2 action send_goal /MoveAbovePlate ada_feeding_msgs/action/MoveTo {}\ `
        2. **Not yet implemented** `ros2 action send_goal /AcquireFood ada_feeding_msgs/action/AcquireFood "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, detected_food: {roi: {x_offset: 0, y_offset: 0, height: 0, width: 0, do_rectify: false}, mask: {header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, format: '', data: []}, item_id: '', confidence: 0.0}}"`
    2. With the web app:
        1. Launch the perception nodes:
            1. Dummy nodes: `TODO`
            2. Real nodes: `TODO`
        2. Launch the web app ([instructions here](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp))

## Writing Behavior Trees That Can Be Wrapped Into Action Servers

### Subclassing `ActionServerBT`

In order to wrap a behavior tree into an action server, you have to subclass `ActionServerBT`. Your subclass must implement the following functions:
- `create_tree`: Create the tree. This is called **once**, sometime before the first call to this action server begins executing.
- `send_goal`: Take a goal from the action server and send it to the tree. This is called **once per goal** that is sent to this action server.
- `preempt_goal`: Take a preempt request from the action server and send it to the tree. This may be called **multiple times per preempt request** that is sent to this action server.
- `was_preempted`: Return whether a preempt request has been fully processed by the tree. This is called **once per action server iteration** after a preempt has been requested and before the preempt has been fully processed.
- `get_feedback`: Return the feedback type that the ROS2 action is expecting, by examining the tree's current state. This is called **once per action server iteration** (which is the same as being called once per `tree.tick()`).
- `get_result`: Return the result type that the ROS2 action is expecting, by examining the tree's current state. This is called **once per goal**, after the tree's status has switched away from `RUNNING`.

Note the following design paradigms when writing this subclass of `ActionServerBT`:
- None of the above functions should be blocking.
- All communication between this file and the behavior tree should take place through the blackboard.
- Only this file should need to import the ROS action type for the action server. The specific behaviors in the tree should not need access to that information.

### Updating the Config File

Then, to execute the behavior tree in `create_action_servers.py`, you have to modify the config file (e.g., `ada_feeding_action_servers.yaml`). The config file has the following structure:

- `action_servers`: A top-level key that contains a list of parameters specifying each action server. Specifically, each element in this list can have the following keys:
    - `server_name` (string, required): the name of the action server.
    - `action_type` (string, required): the ROS action type associated with this action, e.g., `ada_feeding_msgs.action.MoveTo`
    - `tree_class` (string, required): the subclass of `ActionServerBT` that creates the tree and provides functions to interface with the action server, e.g., `ada_feeding.trees.MoveAbovePlate`
    - `tree_kwargs` (dict, optional, default `{}`): a dictionary of keyword arguments to pass to the initialization function of the `tree_class`.
    - `tick_rate` (int, optional, default `30`): the frequency (Hz) at which the action server should tick the tree.

Once you've made the above changes, you should be good to go. Rebuild your workspace and run the launchfile!
