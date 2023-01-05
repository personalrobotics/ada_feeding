#!/usr/bin/env python
# This file launches a ROS node that continuously:
# 1. Gets a detected apriltag frame from apriltag_ros
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np

import tf
from tf.transformations import quaternion_matrix

from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager
from pose_estimators.run_perception_module import run_detection
from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem

def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    if angle < -45.0:
      angle = angle + 90.0
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], angle)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[min(pts[0][1],pts[2][1]):max(pts[0][1],pts[2][1]), 
                       min(pts[0][0],pts[2][0]):max(pts[0][0],pts[2][0])]
    return img_crop

class AprilTagPerception(PoseEstimator):
    def __init__(self):
        # Params
        self.image = None
        self.intrinsic = None
        # Load the tf listener
        self.listener = tf.TransformListener()

    def callback(self, rgb_msg, camera_info):
      # Record Camera Info
      self.intrinsic = np.array(camera_info.K).reshape([3, 3])
      cameraD = np.array(camera_info.D)
      # Record Image
      np_arr = np.frombuffer(rgb_msg.data, dtype=np.uint8)
      np_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
      self.image = cv2.undistort(np_image, self.intrinsic, cameraD)

    def start(self):
      # Image Subscribers
      image_sub = message_filters.Subscriber('image_compressed', CompressedImage)
      info_sub = message_filters.Subscriber('camera_info', CameraInfo)
      ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], 10, 0.2)
      ts.registerCallback(self.callback)

      self.pub = rospy.Publisher('~food_crop', Image, queue_size=1)
      self.pub_compressed = rospy.Publisher('~food_crop/compressed', CompressedImage, queue_size=1)

      # Debugging Only
      #cv2.startWindowThread()
      #cv2.namedWindow("Image")
      #cv2.namedWindow("Crop")

    def detect_objects(self):
      # Load the transform frames
      camera_frame = rospy.get_param("~camera_frame")
      detected_tag_frame = rospy.get_param("~tag_frame")

      # TODO: Move from Hard-Coded to a ROS Param
      corner = np.array([[-0.01825, 0.02175, 0.02175, -0.01825], [-0.035, -0.0345, -0.0595, -0.060], [0.0, 0.0, 0.0, 0.0]])

      transform = None
      try:
          # Get the tag's pose in the camera's frame
          transform = self.listener.lookupTransform(
              camera_frame, detected_tag_frame, rospy.Time(0))
      except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
          return []
      trans, rot = transform
      rot = quaternion_matrix(rot)[:3,:3]
      trans = np.array(trans).reshape((3, 1))
      corner_camera = rot @ corner + trans

      if self.image is None:
        return []

      draw = np.copy(self.image)
      homogen = lambda x: x[:-1]/x[-1]
      uv = homogen((self.intrinsic @ corner_camera)).astype(np.int32).T
      uv = uv.reshape((-1, 1, 2))
      rect = cv2.minAreaRect(uv)
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      crop = crop_minAreaRect(draw, rect)
      if crop.size < 1:
        return []
      # Debugging Only
      # cv2.drawContours(draw,[box],0,(255,0,0),2)
      # cv2.imshow('Image', draw)
      # cv2.imshow('Crop', crop)

      #### Create CompressedImage ####
      msg = CompressedImage()
      msg.header.stamp = rospy.Time.now()
      msg.format = "jpeg"
      msg.data = np.array(cv2.imencode('.jpg', crop)[1]).tostring()
      self.pub.publish(CvBridge().cv2_to_imgmsg(crop))
      self.pub_compressed.publish(msg)
      
      item_center = np.mean(corner, axis=1).reshape((3, 1))
      item_center = rot @ item_center + trans
      pose = np.zeros((4, 4))
      pose[3, 3] = 1.0
      pose[:3, 3] = item_center.flatten()
      pose[:3, :3] = rot

      item = DetectedItem(
            frame_id=camera_frame,
            marker_namespace="food",
            marker_id=1,
            db_key="food_item",
            pose=pose,
            detected_time=rospy.Time.now(),
            info_map=dict())

      return [item]

if __name__ == '__main__':
    rospy.init_node("food_detector")
    my_node = AprilTagPerception()
    my_node.start()

    marker_manager = MarkerManager(count_items=False)

    perception_module = PerceptionModule(
        pose_estimator=my_node,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,
        detection_frame=rospy.get_param("~camera_frame"),
        destination_frame=rospy.get_param("~camera_frame"),
        purge_all_markers_per_update=True)

    destination_frame_marker_topic = rospy.get_name()
    frequency = 30
    run_detection(destination_frame_marker_topic, frequency, perception_module)


