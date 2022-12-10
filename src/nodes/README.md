# Behavior Tree Leaf Nodes

Below is a list of Leaf Nodes for the ADA Feeding project. Decorator and Control nodes should be placed in their own sub-folders.

All vector ports can be represented as a string with semicolon (;) separation between values. All vectors are in `(X,Y,Z)` format, and all quaternions are in `(W,X,Y,Z)` format.

All nodes will include a port list, with each port formatted as: `port_name` {port_type(size), <required/optional/output>): Doc

## Ada Nodes

These are a wrapper around functions specific to the Ada subclass (but not present in the RosRobot class).

* `Talk`: Runs non-blocking text-to-speech with `aoss` on the provided string
  * `say` {std::string, required}

### Hand Interface
* `AdaOpenHand` (No Ports): Executes the `open` preshape to open Ada's Hand.
* `AdaCloseHand` (No Ports): Executes the `close` preshape to open Ada's Hand.
* `AdaHandPreshape`: Executes the named [preshape](https://github.com/personalrobotics/libada/blob/master/resources/g2_6d_named_configs.yaml) on the Ada Hand
  * `preshape` {std::string, required}
* `AdaHandConfig`: Moves Ada's Hand to some specific configuration
  * `config` {std::vector<double>(Num_Hand_DoF), required}

### PostProcessor Interface

* `AdaSetLimits`: Sets the Kunz trajectory post-processor to use the provided motion limits
  * `velocity` {std::vector<double>(Num_Arm_Dofs), optional}: Defaults to the URDF-defined limit *(Note: the default will likely trip the safety detectors in `rewd_controllers`)*
  * `acceleration` {std::vector<double>(Num_Arm_Dofs), optional}: Defaults to the URDF-defined limit
* `AdaGetLimits`: Gets the currently set motion limits
  * `velocity` {std::vector<double>(Num_Arm_Dofs), output}
  * `acceleration` {std::vector<double>(Num_Arm_Dofs, output}

## Config Nodes

These are feeding-specific nodes used to configure the Blackboard Inputs for other nodes.
  
* `ConfigActionSelect`: Determine the action to use to pick up the detected foods.
  * `foods` {std::vector<DetectedObject>(variable), required}: List of detected foods from the Perception module. *Note: currently only the first food is used*
  * `action` {std::vector<double>(variable), output}: Action parameters for other config nodes
* `ConfigMoveAbove`: Prepares the action to move the fork above the detected object.
  * `objects` {std::vector<DetectedObject>(variable), required}: List of detected objects from the Perception module. *Note: currently only the first object is used*
  * `action` {std::vector<double>(variable), required}: Action parameter, currently only understands the 6-action space from SPANet
  * `orig_pos`, `orig_quat`, `pos`, `quat`, `bounds` {std::vector<double>(3, 4, 3, 4, 6), output}: TSR definition for a future `PlanToPose` node
* `ConfigMoveInto`: Prepars the action to move the fork towards an object
  * `objects` {std::vector<DetectedObject>(variable), required}: List of detected objects from the Perception module. *Note: currently only the first object is used*
  * `overshoot` {double, optional}: if provided, calculate an offset that moves the end-effector *past* the detected object (e.g. into the food)
  * `offset` {std::vector<double>(3), output}: end-effector offset, to be used in future `PlanToOffset` node

## Debug Nodes
  
* `Success` (No Ports): Returns SUCCESS
* `Failure` (No Ports): Returns FAILURE
* `Debug` (No Ports): **Blocks the Tree Thread** and opens `cin` prompt for user to select SUCCESS, FAILURE, or RUNNING return

  
## Forque Nodes

Nodes to control the ForceTorque sensor interface (using `FTThresholdHelper`). Note that this will return SUCCESS with no function if Ada is running in simulation mode.

* `SetFTThreshold`: Sets the Force/Torque abort threshold for the running controller.
  * `preset` {std::string, optional}: named F/T thresholds, defined as rosparam `ftSensor/thresholds/<preset/<force/torque>`
  * `force`, `torque` {double, optional}: The thresholds to set. If exceeded, any running trajectory will abort.
  * `retare` {bool, optional}: If true, re-tare the F/T sensor while setting thresholds. Defaults to false.
  * *Note: either `preset` OR (`force` and `torque`) must be set*
  
## Perception Nodes
  
Nodes to interface directly with the `aikido::perception` library. Deales with `aikido::perception::DetectedObject`s.

* `PerceiveFood`, `PerceiveFace`: Wait to detect face/food objects. Reports FAILURE if no objects are returned.
  * `timeout` {double, optional}: if provided, how long to wait for new objects
  * `objects` {std::vector<DetectedObject>(variable), output}: detected objects
  * `name_filter` {std::string, optional}: `PerceiveFood`-only, only return objects with a given name (`DetectedObject::getName`).
* `IsMouthOpen`: If a DetectedObject is a face, return SUCCESS if mouth is reported open. Otherwise FAILURE.
  * `faces` {std::vector<DetectedObject>(variable), required}: *Note: currently only the first face is used*
* `ClearPerceptionList`: Utility node to erase a `DetectedObjedt` vector on the Blackboard
  * `target` {std::vector<DetectedObject>(variable), output}: Outputs an empty list to this Blackboard key.
  
## Robot Nodes
  
These are a wrapper around functions in the `aikido::robot::Robot` and `aikido::robot::RosRobot` classes.
  
* `AdaExecuteTrajectory`: Executes a trajectory output by a Planning node, returns FAILURE if aborted. Adds Trajectory Marker to the InteractiveMarkerViewer.
  * `traj` {aikido::trajectory::TrajectoryPtr, required}
* Documentation for ports shared between all planning functions (i.e. `AdaPlanToX`)
  * `worldCollision` {bool, optional}: Whether to use the World collision constraint. If false, defaults to self-collision only. Defaults to false.
  * `traj` {aikido::trajectory::TrajectoryPtr, output}: If SUCCESS, contains a valid trajectory to execute
* `AdaPlanToOffset`: Wrapper for the VectorFieldPlanner. Plans an end-effector motion to the specified translation offset in world frame.
  * `offset` {std::vector<double>(3), required}
* `AdaPlanToPoseOffset`: Wrapper for the VectorFieldPlanner. Plans an end-effector motion to a pose specified by a translaiton and rotation offset.
  * `offset` {std::vector<double>(3), required}
  * `rotation` {std::vector<double>(3), required}
* `AdaPlanToConfig`: Wrapper for basic configuration planner. Plans the robot to a certain position in joint configuration space.
  * `config` {std::vector<double>(Num_Arm_Dofs), required}
* `AdaPlanToPose`: Wrapper for Task Space Region (i.e. TSR) planner. Samples points in provided TSR, runs IK, and plans to the resulting joint configuration.
  * `orig_pos`, `orig_quat` {std::vector<double>(3, 4), optional}: `Tw_0`, i.e., planning frame relative to world origin. Defaults to identity.
  * `pos`, `quat` {std::vector<double>(3, 4), optional}: `Te_w`, i.e., end-effector position relative to the planning frame origin. Defaults to 0.
  * `bounds` {std::vector<double>(6), optional}: Defines the size/shape of the TSR by specifying bounds of the *planning frame origin*. Represented as `(X,Y,Z,Roll,Pitch,Yaw)`. Defaults to 0 (i.e. single pose).
  * `lbounds` {std::vector<double>(6), optional}: If defined, creates asymmetric bounds for the TSR. Defaults to `-bounds`.
  * `viz` {bool, optional}: If true, visualizes samples of the TSR in the InteractiveMarkerViewer until cleared. Useful for debugging. Default to false.
* `AdaGetEEPose`: Gets the position of Ada's End Effector in the world frame.
  * `pos`, `quat` {std::vector<double>(3, 4), output}
* `AdaGetConfig`: Gets the joint configuration of the robot
  * `armOnly` {bool, optional}: if true, ignore the gripper. Defaults to true.
  * `target` {std::vector<double>(Num_Arm_Dofs OR Num_Dofs), output}

## ROS Nodes

These are wrappers around common ROS operations.
  
* `RosSubX`: Subscribe to a `std_msgs` topic and write the first new result to the Blackboard.
  * `topic` {std::string, required}: Ros Topic, passed directly to `NodeHandle::subscribe`
  * `target` {X, output}: Type determined by implementation
  * Implemented types:
    * `RosSubString` (std_msgs::String -> std::string)
    * `RosSubBool` (std_msgs::Bool -> bool)
    * `RosSubD` (std_msgs::Float64 -> double)
    * `RosSubI` (std_msgs::Int32 -> int)
    * `RosSubVecD` (std_msgs::Float64MultiArray -> std::vector<double>): 1D array only
    * `RosSubVecI` (std::msgs::Int32MultiArray -> std::vector<int>): 1D array only
* `RosGetX`: Load a rosparam into the Blackboard, returns FAILURE if no value is present or default provided
  * `param` {std::string, required}: Rosparam key, passed directly to `NodeHandle::getParam`
  * `default` {X, optional}: Default value to use if param is not in server
  * `target` {X, optional}: Type determined by implementation
  * Implemented Types: `RosGetString` (std::string), `RosGetBool` (bool), `RosGetD` (double), `RosGetI` (int), `RosGetVecD` (std::vector<double>), `RosGetVecI` (std::vector<int>)
* `RosSetX`: Writes rosparam from the Blackboard
  * `param` {std::string, required}: Rosparam key, passed directly to `NodeHandle::setParam`
  * `value` {X, optional}: Type determined by implementation
  * Implemented Types: `RosSetString` (std::string), `RosSetBool` (bool), `RosSetD` (double), `RosSetI` (int), `RosSetVecD` (std::vector<double>), `RosSetVecI` (std::vector<int>)

  
## Safety Nodes
  
These are nodes used to handle critical safety-oriented user interaction.
  
* `CheckWatchdog` (No Ports): returns FAILURE if the watchdog hasn't been fed (i.e. received `true`) recently.
  * *Note: Will return RUNNING if the watchdog has never been fed since start-up*
  * *Note: "recently" is the `watchdogTimeout` defined by rosparameter*
  * *Note: Listens to the `~watchdog` topic, expecting std_msgs::Bool*
  
## World Nodes
  
These are used to interact with the Aikido World object used by the Robot object.

* `WorldAddUpdate`: Adds a skeleton to the World object
  * `skelName` {std::string, required}: Name of object in world. If present, update. If not, Add.
  * `urdfUri` {std::string, optional}: Path to object URDF file, required if Add operation.
  * `pos`, `quat` {std::vector<double>(3, 4), optional}: Transform to add/move object to relative to world origin. Defaults to Identity.
* `WorldRemove`: Removes a skeleton from the World. Cannot remove Ada.
  * `skelName` {std::string, required}
* `WorldClearFrames` (No Ports): Removes frame markers (e.g. TSR Markers) from the InteractiveMarkerViewer.
* `WorldClearTraj` (No Ports): Removes trajectory markers from the InteractiveMarkerViewer.
  
## Creating New Nodes

* Create a new cpp file in `src/nodes`
* Add the CPP file to CMakeLists sources under `file(GLOB_RECURSE SOURCES ...`
* Don't forget to format the code:
  * `cd ada_feeding; mkdir build`
  * `cd build; cmake ..`
  * `make format`
* Put the main function code for Node initialization in `registerNodes()`, see Boilerplate.
* Nodes will be registered automatically by `nodes.hpp` on program start.

### Boilerplate
```
// Header to general Node functionality
#include "feeding/nodes.hpp"

namespace feeding {
namespace nodes {

// ... Add Node Functions / classes / etc.

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada &robot) {
    // Add node initialization here
    // This is called by the main() function in main.cpp
}
// This is called a static{}, i.e., before main() is called
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
```

### Best Practices

* **Do Not Block the Tree Thread**. All nodes should use e.g. `std::async` to return a status to the tree ASAP.
* Nodes that rely on shared state (e.g. the [GripperInterface](https://www.behaviortree.dev/docs/tutorial-basics/tutorial_01_first_tree)) can be defined in a `static` object.
* Do not pollute global scope (*Note: the InteractiveMarkerViewer is an exception to prevent race conditions.*)
* In general, it is best to pass the NodeHandle by reference (using `std::ref` in `std::bind`), and the Ada object by bare pointer.
* Note that any *static* constructors are run before the ROS node is created. So the NodeHandle must be stored as a bare pointer and initialized with a separate `init()` function that runs in `registerNodes()`.
* Bare pointers to the NodeHandle and ADA object should be safe to use for the lifetime of the program. **DO NOT PERFORM ANY MEMORY OPERATIONS [e.g. free/delete] ON THESE POINTERS**
  
## Node TODOs

* ROS Publisher Nodes
* ROS state service node (i.e. allow other nodes to query some String state)
* Executor/controller management (once merged into Aikido)
* Camera interaction nodes (i.e. pull a frame / camera info from the camera topic)
* Generic ROS service interaction client nodes (use templates to remove repeated logic)
* Augment `Talk` to block with `BT::NodeStatus::RUNNING`
* Forque nodes to collect raw F/T values and their moments (mean, std, etc.)
* Remove code reuse in AdaPerception object: replace `m<Food/Face>Detector` with map from `string` (i.e. the RosTopic) to PoseEstimatorModule. Have `EnablePerception` loop to create Timer with given RosTopic string, and `Perceive` that takes in the given RosTopic. 
* Change CheckWatchdog to generic node where you can pass a topic name to listen to. Requires the static EStopInterface to have a map of topics to subscribers. Useful to separate Watchdog and Bite Cancellation actions.
* TrajectoryMarker in `AdaExecuteTrajectory` uses the last arm node instead of the end-effector. That should be fixed. See .cpp file for more info.
