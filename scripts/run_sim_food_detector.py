#!/usr/bin/python
# Example script for running a perception module

from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager
from pose_estimators.run_perception_module import run_detection
from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem
import rospy
import numpy as np


# This script runs a sim food detector which publishes markers for food items
# on the plate.

class SimFoodDetector(PoseEstimator):
    def __init__(self, frame_id):

        # Pose at which the food is on the plate
        pose1 = np.array([[1, 0, 0, 0.30],
                          [0, 1, 0, -0.25],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item1 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=1,
            db_key="food_item",
            pose=pose1,
            detected_time=rospy.Time.now(),
            info_map=dict(action="tilted-vertical", rotation=0.0, score=1.0, annotation='tv'))

        # Pose at which the food is on the plate
        pose2 = np.array([[1, 0, 0, 0.25],
                          [0, 1, 0, -0.29],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item2 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="grape",
            marker_id=1,
            db_key="food_item",
            pose=pose2,
            detected_time=rospy.Time.now(),
            info_map=dict(action="vertical", rotation=90.0, score=1.0))

    def detect_objects(self):
        self.item1.detected_time = rospy.Time.now()
        self.item2.detected_time = rospy.Time.now()
        return [self.item1, self.item2]


# When running without a robot, publish a static transform between map and another frame
# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 map base_frame 1000
# You should be able to see marker array in rviz under topic /food_detector/marker_array
if __name__ == "__main__":
    detection_frame = "map"
    destination_frame = "map"

    rospy.init_node("food_detector")

    pose_estimator = SimFoodDetector(detection_frame)
    marker_manager = MarkerManager(count_items=False)

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,
        detection_frame=detection_frame,
        destination_frame=destination_frame,
        purge_all_markers_per_update=True)

    destination_frame_marker_topic = rospy.get_name()
    frequency = 5
    run_detection(destination_frame_marker_topic, frequency, perception_module)
