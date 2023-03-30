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
    def __init__(self, frame_list):

       self.frame_list = frame_list

    def detect_objects(self):
        ret = []
        for frame in self.frame_list:
            item = DetectedItem(
                frame_id=frame,
                marker_namespace=frame,
                marker_id=1,
                db_key="food_item",
                pose=np.eye(4),
                detected_time=rospy.Time.now(),
                info_map={})
            ret.append(item)
        return ret


# When running without a robot, publish a static transform between map and another frame
# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 map base_frame 1000
# You should be able to see marker array in rviz under topic /food_detector/marker_array
if __name__ == "__main__":
    detection_frame = "map"
    destination_frame = "map"

    rospy.init_node("food_detector")
    frame_list = rospy.get_param("~frame_list")

    pose_estimator = SimFoodDetector(frame_list)
    marker_manager = MarkerManager(count_items=False)

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        purge_all_markers_per_update=True)

    destination_frame_marker_topic = rospy.get_name()
    frequency = 5
    run_detection(destination_frame_marker_topic, frequency, perception_module)
