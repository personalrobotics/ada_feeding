from rclpy.node import Node
from sensor_msgs.msg import JointState
import py_trees

from ada_feeding.behaviors import BlackboardBehavior

class GetJointStates(BlackboardBehavior):
    def __init__(self, name, joint_names, output_key, ns="/"):
        super().__init__(name, ns=ns, inputs={"joint_names": joint_names}, outputs={"joint_states": output_key})
        self.ns = ns
        self.node = None
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
        self.node.get_logger().info("Joint states callback received")
        joint_names = self.blackboard_get("joint_names")
        for i, name in enumerate(msg.name):
            if name in joint_names:
                self.node.get_logger().info(f"Storing joint state: {name} = {msg.position[i]}")
                self.latest_joint_states[name] = msg.position[i]
        self.node.get_logger().info(f"Joint names in message: {msg.name}")

        # Check if we have received all the required joints
        if all(name in self.latest_joint_states for name in self.blackboard_get("joint_names")):
            self.received_all_joints = True
            self.node.get_logger().info("Received all joints!")

    def update(self):
        self.node.get_logger().info("GetJointStates.update() called")
        self.node.get_logger().info(f"Current joint states: {self.latest_joint_states}")
        # Check if we have the joint states we need
        if self.received_all_joints:
            self.blackboard_set("joint_states", self.latest_joint_states)
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.joint_states_sub = None
        self.latest_joint_states.clear()
        self.received_all_joints = False
