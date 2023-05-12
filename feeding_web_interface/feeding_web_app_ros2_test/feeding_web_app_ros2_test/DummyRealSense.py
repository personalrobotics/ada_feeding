#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge
import pathlib
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import threading


class DummyRealSense(Node):
    """
    Reads in a video file and publishes the frames as ROS2 messages.
    """

    def __init__(self, video_path, fps=30, topic="/camera/color/image_raw"):
        super().__init__("dummy_real_sense")

        # Read in the video file
        self.video = cv2.VideoCapture(video_path)
        self.fps = fps

        # Create the publisher
        self.topic = topic
        self.publisher_ = self.create_publisher(Image, topic, 1)
        self.num_frames = 0
        self.bridge = CvBridge()

        # Launch the publisher in a separate thread
        self.thread = threading.Thread(target=self.publish_frames, daemon=True)
        self.thread.start()

    def publish_frames(self):
        # Maintain the rate
        rate = self.create_rate(self.fps)

        while True:
            # Capture frame-by-frame
            ret, frame = self.video.read()
            self.num_frames += 1
            # If the last frame is reached, reset the capture and the frame_counter
            if self.num_frames == self.video.get(cv2.CAP_PROP_FRAME_COUNT):
                self.num_frames = 0
                self.video.set(cv2.CAP_PROP_POS_FRAMES, self.num_frames)
            # Convert to a ROS2 msg
            frame_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            frame_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(frame_msg)
            # # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # # Display the resulting frame
            # cv2.imshow('frame',gray)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            rate.sleep()


def main(args=None):
    rclpy.init(args=args)

    video_name = "2022_11_01_ada_picks_up_carrots_camera_compressed_ft_tf.mp4"
    video_path = str(
        (
            pathlib.Path(__file__).parent.parent.parent.parent.parent
            / "share/data"
            / video_name
        ).resolve()
    )

    dummy_real_sense = DummyRealSense(video_path)

    rclpy.spin(dummy_real_sense)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dummy_real_sense.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
