# ROS Client Library for the Python language
import rclpy
# Create a Node
from rclpy.node import Node
# A representation of pose in free space, composed of position and orientation
from geometry_msgs.msg import Pose
# Threading in python is used to run multiple threads (tasks, function calls) at the same time.
import threading

# main function with no arguments passed
def main(args=None):
    # initialize node with args parameter
    rclpy.init(args=args)
    #  instantiates the ROS node you are implementing.
    node = Node("get_eef_pose")
    # instantiation and invocation of the Single-Threaded Executor, which is the simplest Executor
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin)
    spinner.start()
    
    # use the group name
    move_group_interface = moveit_commander.MoveGroupCommander(kinova.MOVE_GROUP_ARM)
    
    # get current pose for the move_group_interface
    current_pose = move_group_interface.get_current_pose().pose

    node.get_logger().info("Current pose: %f %f %f %f %f %f %f",
        current_pose.position.x,
        current_pose.position.y,
        current_pose.position.z,
        current_pose.orientation.x,
        current_pose.orientation.y,
        current_pose.orientation.z,
        current_pose.orientation.w)

    rclpy.shutdown()
    spinner.join()
    return 0

if __name__ == '__main__':
    main()
