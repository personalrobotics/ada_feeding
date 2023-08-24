#!/usr/bin/env python3
import math

from geometry_msgs.msg import Pose

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ada_feeding.srv import Move


class FrameListener(Node):

    def __init__(self):
        super().__init__('gripper_tf2_frame_listener')

        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
          'target_frame', 'world').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create a client to move the gripper
        self.mover = self.create_client(Move, 'move')
        # Boolean values to store the information
        # if the service for moving gripper center is available
        self.gripper_moving_service_ready = False
        # if the gripper was successfully moved
        self.gripper_moved = False

        # Create gripper position publisher
        # declares that the node publishes message type,
        # a topic name, and queue size that limits the amount 
        # of queued messages if a subscriber is not receiving them fast enough.
        self.gripper = self.create_publisher(Pose, 'gripper/xy_position', 1)

        # Call on_timer function every second
        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = self.target_frame
        to_frame_rel = 'j2n6s200_end_effector'

        if self.gripper_moving_service_ready:
            if self.gripper_moved:
                # Look up for the transformation between target_frame and gripper center frames
                # and send position commands for gripper center to reach target_frame
                try:
                    t = self.tf_buffer.lookup_transform(
                        to_frame_rel,
                        from_frame_rel,
                        rclpy.time.Time())
                except TransformException as ex:
                    self.get_logger().info(
                        f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                    return

                msg = Pose()
                scale_movement_rate = 0.001
                # move left: - 0.001 + x, move right: 0.001 + x
                # move forward: 0.001 + y, move backward: -0.001 + y
                msg.position.x = scale_rotation_rate + t.transform.translation.x
                self.publisher.publish(msg)
            else:
                if self.result.done():
                    self.get_logger().info(
                        f'Successfully moved {self.result.result().name}')
                    self.gripper_moved = True
                else:
                    self.get_logger().info('Movement is not finished')
        else:
            if self.mover.service_is_ready():
                # Initialize request with turtle name and coordinates
                # Note that x, y and theta are defined as floats in turtlesim/srv/Move
                # request = Move.Request()
                request.name = 'j2n6s200_end_effector'
                request.x = float(4)
                request.y = float(4)
                # Call request
                self.result = self.mover.call_async(request)
                self.gripper_moving_service_ready = True
            else:
                # Check if the service is ready
                self.get_logger().info('Service is not ready')


def main():
    rclpy.init()
    node = FrameListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()