#!/usr/bin/env python3
import os
import pprint
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import traceback
import yaml


class CreateActionServers(Node):
    """
    The CreateActionServers node initializes a series of action servers, each
    of which runs a behavior tree (implemented using the py_trees library).
    The mapping between behavior tree files and the parameters of an action
    server are specified in a configuration file.

    This node ensures that only one of the action servers is executing a goal
    at any given time. Consequently, any goal requests received (to any of the
    action servers exposed by this node) while an action server already has a
    goal will be rejected. This is useful e.g., in the case where each action
    servers command a robot to move in a particular way; since the robot can
    only execute one motion at once, all other action servers must reject
    incoming goal requests.
    """

    def __init__(self):
        """
        Initialize the CreateActionServers node. This function reads in the 
        configuration file and creates the action servers. Note that it does not
        load the behavior tree file associated with each action server; that
        happens when the action server receives a goal request.
        """
        super().__init__('create_action_servers')

        # Declare the ROS2 parameters this node will be looking up.
        self.declare_parameter('config_file', rclpy.Parameter.Type.STRING)

        # Get the path to the configuration file.
        config_file = self.get_parameter('config_file').value
        if (not os.path.isfile(config_file)):
            raise Exception('Path specified in config_file parameter does not exist: %s' % config_file)
        
        # Read the configuration file.
        self.get_logger().info('Loading configuration file: %s' % config_file)
        configs = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        self.get_logger().debug("\n"+pprint.pformat(configs, indent=4))

        # For every action server specified in the configuration file, create
        # an action server.
        self._action_servers = []
        self._action_types = {}
        for action_config in configs["action_servers"]:
            # Load the action server parameters from the configuration file.
            server_name = action_config["server_name"]
            action_type = action_config["action_type"]
            self.get_logger().info('Creating action server %s with type %s' % (server_name, action_type))

            # Import the action type
            try:
                action_package, action_module, action_class = action_type.split("/", 3)
            except Exception as e:
                raise Exception('Invalid action type %s. Except "package/module/class" e.g., "ada_feeding_msgs/action/MoveTo"' % action_type)
            try:
                self._action_types[action_type] = getattr(getattr(__import__('%s.%s' % (action_package, action_module)), action_module), action_class)
            except Exception as e:
                raise Exception('Error importing %s: %s' % (action_type, e))

            # Create the action server.
            action_server = ActionServer(
                self,
                self._action_types[action_type],
                server_name,
                self.get_execute_callback(server_name, action_type),
            )
            self._action_servers.append(action_server)

    def get_execute_callback(self, server_name, action_type):
        """
        This is a wrapper function that takes in information about the action
        server and returns a callback function that can be used for that server.
        This is necessary because the callback function must return a Result and
        Feedback consisitent with the action type.
        """
        async def execute_callback(goal_handle):
            """
            This function is called when a goal is accepted by an action server.
            It loads the behavior tree file associated with the action server
            (if not already loaded) and executes the behavior tree, publishing
            periodic feedback.
            """
            self.get_logger().info(
                "%s: Executing goal %s with request %s" % (server_name, ''.join(format(x, '02x') for x in goal_handle.goal_id.uuid), goal_handle.request)
            )
            goal_handle.succeed()
            return self._action_types[action_type].Result()

        return execute_callback

def main(args=None):
    rclpy.init(args=args)

    create_action_servers = CreateActionServers()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(create_action_servers, executor=executor)
    except Exception as e:
        traceback.print_exc()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    create_action_servers.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()