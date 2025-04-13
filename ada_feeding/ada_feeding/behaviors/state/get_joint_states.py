from rclpy.node import Node
from sensor_msgs.msg import JointState
import py_trees

from typing import Any, Dict, List, Optional, Union
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey

class GetJointStates(BlackboardBehavior):
    def __init__(self, name: str, node=None, ns:str="/", inputs: Optional[Dict[str, Union[BlackboardKey, Any]]] = None, outputs: Optional[Dict[str, Optional[BlackboardKey]]] = None,
):
        super().__init__(name, ns=ns, inputs=inputs, outputs=outputs)
        self.ns = ns
        self.node = node
        self.joint_states_sub = None
        self.latest_joint_states = {}
        self.received_all_joints = False

    def blackboard_inputs(
        self,
        joint_names: Union[BlackboardKey, Optional[List[str]]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        constraints: previous set of constraints to append to
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        joint_positions: Union[BlackboardKey, List[float]] = None,
        joint_names: Union[BlackboardKey, Optional[List[str]]] = None,
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        constraints: list of constraints to send to MoveIt2Plan
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        # Initialize ROS 2 node
        if self.node is None:
            self.node = Node(self.name + "_node", namespace=self.ns)
        # Subscribe to JointState topic
        self.joint_states_sub = self.node.create_subscription(
            JointState, "/joint_states", self.joint_states_callback, 10
        )

    def joint_states_callback(self, msg: JointState):
        joint_names = self.blackboard_get("joint_names")
        for i, name in enumerate(msg.name):
            if name in joint_names:
                self.latest_joint_states[name] = msg.position[i]

    def update(self):
        # Check if we have the joint states we need
        if all(name in self.latest_joint_states for name in self.blackboard_get("joint_names")):
            joint_names = list(self.latest_joint_states.keys())
            joint_positions = list(self.latest_joint_states.values())
            self.node.get_logger().info(f"Received all joint states")
            self.node.get_logger().info(f"Joint Names: {joint_names}")
            self.node.get_logger().info(f"Joint Positions: {joint_positions}")
            self.blackboard_set("joint_names", joint_names)
            self.blackboard_set("joint_positions", joint_positions)
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self.node is not None and self.node != self.node:
            self.node.destroy_node()
            self.node = None
        self.joint_states_sub = None
        self.latest_joint_states.clear()
