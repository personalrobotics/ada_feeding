from rclpy.node import Node
from sensor_msgs.msg import JointState
import py_trees

from ada_feeding.behaviors import BlackboardBehavior

class GetJointStates(BlackboardBehavior):
    def __init__(self, name, joint_names, output_key, node=None, ns="/"):
        super().__init__(name, ns=ns, inputs={"joint_names": joint_names}, outputs={"joint_states": output_key})
        self.ns = ns
        self.node = node
        self.joint_states_sub = None
        self.latest_joint_states = {}
        self.received_all_joints = False

    def setup(self, **kwargs):
        # Initialize ROS 2 node
        if self.node is None:
            self.node = Node(self.name + "_node", namespace=self.ns)
        # Subscribe to JointState topic
        self.joint_states_sub = self.node.create_subscription(
            JointState, "/joint_states", self.joint_states_callback, 10
        )

    def joint_states_callback(self, msg: JointState):
        # self.node.get_logger().info("Joint states callback received")
        joint_names = self.blackboard_get("joint_names")
        for i, name in enumerate(msg.name):
            if name in joint_names:
                # self.node.get_logger().info(f"Storing joint state: {name} = {msg.position[i]}")
                self.latest_joint_states[name] = msg.position[i]

    def update(self):
        # Check if we have the joint states we need
        if all(name in self.latest_joint_states for name in self.blackboard_get("joint_names")):
            self.node.get_logger().info(f"Received all joint states")
            self.blackboard_set("joint_states", self.latest_joint_states)
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self.node is not None and self.node != self.node:
            self.node.destroy_node()
            self.node = None
        self.joint_states_sub = None
        self.latest_joint_states.clear()
