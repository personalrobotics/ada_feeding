#!/usr/bin/python
# Example script for running a perception module

from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager
from pose_estimators.run_perception_module import run_detection

from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem
import rospy
import numpy as np


# This script runs a sim face detector which publishes markers for a
# hypothetical user face with a perpetually open mouth.

class SimFaceDetector(PoseEstimator):
    def __init__(self, frame_id):

        personPose = np.array([
            [0, 1, 0, 0.274968],
            [-1, 0, 0, 0.350025],
            [0, 0, 1, 0.752],
            [0, 0, 0, 1]])

        self.item = DetectedItem(
            frame_id=frame_id,
            marker_namespace="mouth",
            marker_id=1,
            db_key="mouth",
            pose=personPose,
            detected_time=rospy.Time.now(),
            info_map=dict(**{"mouth-status": "open"}))

    def detect_objects(self):
        self.item.detected_time = rospy.Time.now()
        return [self.item]


# Run in command line a static transform between the detection frame to
# destination frame, and map to destination frame
# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 map base_frame 1000
# You should be able to see marker array in rviz under topic /face_pose/marker_array

if __name__ == "__main__":
    detection_frame = "map"
    # Change to Robot Base Link, e.g.:
    destination_frame = "map"

    rospy.init_node("face_pose")

    pose_estimator = SimFaceDetector(detection_frame)
    marker_manager = MarkerManager(count_items=False)

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,  # Not used since pose estimator is provided.
        detection_frame=detection_frame,
        destination_frame=destination_frame,
        purge_all_markers_per_update=True)

    destination_frame_marker_topic = rospy.get_name()
    frequency = 5
    run_detection(destination_frame_marker_topic, frequency, perception_module)
