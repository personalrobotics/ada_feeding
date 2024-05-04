#!/usr/bin/env python3
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_srvs.srv import SetBool
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from threading import Lock


class TableDetectionNode(Node):
    DEFAULT_POSE_STAMPED = PoseStamped(
        header=Header(
            stamp=Time(sec=0, nanosec=0),
            frame_id="root",
        ),
        # NOTE: This pose should be the same as the default pose in
        # `ada_planning_scene.yaml`
        pose=Pose(
            position=Point(x=0.08, y=-0.5, z=-0.48),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )

    def __init__(self):
        """
        Initializes the TableDetectionNode.
        """
        super().__init__("table_detection")

        # Keeps track of whether table detection is on or not
        self.is_on = False
        self.is_on_lock = Lock()

        # Create the service
        self.srv = self.create_service(
            SetBool,
            "toggle_table_detection",
            self.toggle_table_detection_callback,
        )

        # Subscribe to the camera feed
        self.subscription = self.create_subscription(
            CompressedImage,
            "camera/color/image_raw/compressed",
            self.camera_callback,
            1,
        )

        # Create the publishers
        self.publisher_results = self.create_publisher(
            PoseStamped, "table_detection", 1
        )

    def toggle_table_detection_callback(self, request, response):
        """
        Callback function for the SetBool service. Safely toggles
        the table detection on or off depending on the request.
        """
        self.get_logger().info("Incoming service request. turn_on: %s" % (request.data))
        if request.data:
            # Turn on table detection
            self.is_on_lock.acquire()
            self.is_on = True
            self.is_on_lock.release()
            response.success = True
            response.message = "Succesfully turned table detection on"
        else:
            self.is_on_lock.acquire()
            self.is_on = False
            self.is_on_lock.release()
            response.success = True
            response.message = "Succesfully turned table detection off"
        return response

    def camera_callback(self, msg):
        """
        Callback function for the camera feed. If table detection is on, this
        function will detect the table in the image and publish information about
        them to the /table_detection topic.
        """
        self.get_logger().debug("Received image")
        self.is_on_lock.acquire()
        is_on = self.is_on
        self.is_on_lock.release()
        if is_on:
            TableDetectionNode.DEFAULT_POSE_STAMPED.header.stamp = msg.header.stamp
            self.publisher_results.publish(TableDetectionNode.DEFAULT_POSE_STAMPED)


def main(args=None):
    rclpy.init(args=args)

    table_detection = TableDetectionNode()

    rclpy.spin(table_detection)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
