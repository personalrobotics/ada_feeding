import os, sys
import cv2
import numpy as np
import time
import math
import signal

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros

from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float64MultiArray
from types import SimpleNamespace

from threading import Lock
from copy import deepcopy

from scipy.spatial.transform import Rotation

class DualCameraCalibration:
    def __init__(self):
        rospy.init_node('DualCameraCalibration')

        self.top_camera_lock = Lock()
        self.top_camera_color_data = None
        self.top_camera_info_data = None
        self.top_camera_depth_data = None

        self.bottom_camera_lock = Lock()
        self.bottom_camera_color_data = None
        self.bottom_camera_info_data = None
        self.bottom_camera_depth_data = None

        self.color_image_sub = message_filters.Subscriber(rospy.get_param('/head_perception/colorTopicName'), Image)
        self.camera_info_sub = message_filters.Subscriber(rospy.get_param('/head_perception/infoTopicName'), CameraInfo)
        self.depth_image_sub = message_filters.Subscriber(rospy.get_param('/head_perception/depthTopicName'), Image)
        ts = message_filters.TimeSynchronizer([self.color_image_sub, self.camera_info_sub, self.depth_image_sub], 1)
        ts.registerCallback(self.rgbdCallback)

        self.bottom_color_image_sub = message_filters.Subscriber(rospy.get_param('/head_perception/bottomColorTopicName'), Image)
        self.bottom_camera_info_sub = message_filters.Subscriber(rospy.get_param('/head_perception/bottomInfoTopicName'), CameraInfo)
        self.bottom_depth_image_sub = message_filters.Subscriber(rospy.get_param('/head_perception/bottomDepthTopicName'), Image)
        ts = message_filters.TimeSynchronizer([self.bottom_color_image_sub, self.bottom_camera_info_sub, self.bottom_depth_image_sub], 1)
        ts.registerCallback(self.bottomRgbdCallback)

        self.bridge = CvBridge()

        self.top_voxel_publisher =  rospy.Publisher("/dual_camera_calibration/top_voxels/marker_array", MarkerArray, queue_size=10)
        self.bottom_voxel_publisher =  rospy.Publisher("/dual_camera_calibration/bottom_voxels/marker_array", MarkerArray, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.found_transform = None
        self.min_rmsd = 1

    # Any reason that the callback shouldn't be computation intensive?
    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):

        # global viz_image
        time_a = time.time()

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        with self.top_camera_lock:
            self.top_camera_color_data = rgb_image
            self.top_camera_info_data = camera_info_msg
            self.top_camera_depth_data = depth_image

    def bottomRgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        with self.bottom_camera_lock:
            self.bottom_camera_color_data = rgb_image
            self.bottom_camera_info_data = camera_info_msg
            self.bottom_camera_depth_data = depth_image

    def visualizeTopVoxels(self, voxels):

        # print(voxels)

        markerArray = MarkerArray()

        marker = Marker()
        marker.header.seq = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.ns = "visualize_voxels"
        marker.id =  1
        marker.type = 6; # CUBE LIST
        marker.action = 0; # ADD
        marker.lifetime = rospy.Duration()
        marker.scale.x = 0.005
        marker.scale.y = 0.005
        marker.scale.z = 0.005
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1

        for i in range(voxels.shape[0]):

            point = Point()
            point.x = voxels[i,0];
            point.y = voxels[i,1];
            point.z = voxels[i,2];

            marker.points.append(point)
            
        markerArray.markers.append(marker)

        self.top_voxel_publisher.publish(markerArray)

    def visualizeBottomVoxels(self, voxels):

        # print(voxels)

        markerArray = MarkerArray()

        marker = Marker()
        marker.header.seq = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "bottom_camera_color_optical_frame"
        marker.ns = "visualize_voxels"
        marker.id =  1
        marker.type = 6; # CUBE LIST
        marker.action = 0; # ADD
        marker.lifetime = rospy.Duration()
        marker.scale.x = 0.005
        marker.scale.y = 0.005
        marker.scale.z = 0.005
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1

        for i in range(voxels.shape[0]):

            point = Point()
            point.x = voxels[i,0];
            point.y = voxels[i,1];
            point.z = voxels[i,2];

            marker.points.append(point)
            
        markerArray.markers.append(marker)

        self.bottom_voxel_publisher.publish(markerArray)

    def pixel2World(self, camera_info, image_x, image_y, depth_image):

        # print("(image_y,image_x): ",image_y,image_x)
        # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

        if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
            return False, None

        depth = depth_image[image_y, image_x]

        if math.isnan(depth) or depth < 0.05 or depth > 1.0:

            depth = []
            for i in range(-3,3):
                for j in range(-3,3):
                    if image_y+i >= depth_image.shape[0] or image_x+j >= depth_image.shape[1]:
                        return False, None
                    pixel_depth = depth_image[image_y+i, image_x+j]
                    if not (math.isnan(pixel_depth) or pixel_depth < 50 or pixel_depth > 1000):
                        depth += [pixel_depth]

            if len(depth) == 0:
                return False, None

            depth = np.mean(np.array(depth))

        depth = depth/1000.0 # Convert from mm to m

        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]  

        # Convert to world space
        world_x = (depth / fx) * (image_x - cx)
        world_y = (depth / fy) * (image_y - cy)
        world_z = depth

        return True, [world_x, world_y, world_z]

    def kabschUmeyamaNoScale(self, A, B):

        assert A.shape == B.shape

        # Calculate translation using centroids
        A_centered = A - np.mean(A, axis=0)
        B_centered = B - np.mean(B, axis=0)

        R, rmsd = Rotation.align_vectors(A_centered, B_centered)
        print("RMSD: ",rmsd)

        t = np.mean(A, axis=0) - R.as_matrix()@np.mean(B, axis=0)

        return rmsd, R.as_matrix(), t.squeeze()

    def detect_ar_tags_corners(self, image):

        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        # corners = np.array(corners).reshape(-1,2)

        corners_ordered = []

        corners_with_ids = []
        if len(corners) > 0:
            # print("Detected arUco")

            ids = ids.flatten()

            for (markerCorner, markerID) in zip(corners, ids):
                corners_with_ids.append((markerID,markerCorner.tolist()))

            corners_with_ids.sort()

            for (markerID, markerCorner) in corners_with_ids:
                corners_ordered.append(markerCorner)


        # print("corners_ordered: ", corners_ordered)
        # print("ids: ",ids)

        # lol = input()

        # print("corners: ",corners)
        # print("corners.shape: ",corners.shape)

        corners_ordered = np.array(corners_ordered).reshape(-1,2)

        return corners_ordered

    def get_transformation_from_tf(self):

        while not rospy.is_shutdown():
            try:
                transform = self.tfBuffer.lookup_transform('base_link', 'camera_link', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("Looking for transform")
                continue

        base_to_camera = np.zeros((4,4))
        base_to_camera[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        base_to_camera[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        base_to_camera[3,3] = 1

        while not rospy.is_shutdown():
            try:
                transform = self.tfBuffer.lookup_transform('base_link', 'bottom_camera_link', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("Looking for transform")
                continue

        bottom_base_to_camera = np.zeros((4,4))
        bottom_base_to_camera[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        bottom_base_to_camera[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        bottom_base_to_camera[3,3] = 1

        top_to_bottom = np.linalg.inv(base_to_camera) @ bottom_base_to_camera

        return top_to_bottom

    def get_transformation_from_ar_tags(self):

        while not rospy.is_shutdown():
            try:
                transform = self.tfBuffer.lookup_transform('camera_link', 'camera_color_optical_frame', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("Looking for transform ... ")
                continue

        top_cc = np.zeros((4,4))
        top_cc[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        top_cc[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        top_cc[3,3] = 1

        while not rospy.is_shutdown():
            try:
                transform = self.tfBuffer.lookup_transform('bottom_camera_link', 'bottom_camera_color_optical_frame', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("Looking for transform ... ")
                continue

        bottom_cc = np.zeros((4,4))
        bottom_cc[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        bottom_cc[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        bottom_cc[3,3] = 1

        # print("cc: ",cc)
        # lol = input()

        top_camera_color_data = None
        top_camera_info_data = None
        top_camera_depth_data = None
        bottom_camera_color_data = None
        bottom_camera_info_data = None
        bottom_camera_depth_data = None

        last_time = time.time()
        rate = rospy.Rate(1.0)
        count = 0
        while True:

            count += 1
            print("Frequency: ",1.0/(time.time() - last_time))
            last_time = time.time()

            with self.top_camera_lock:
                top_camera_color_data = deepcopy(self.top_camera_color_data)
                top_camera_info_data = deepcopy(self.top_camera_info_data)
                top_camera_depth_data = deepcopy(self.top_camera_depth_data)
            with self.bottom_camera_lock:
                bottom_camera_color_data = deepcopy(self.bottom_camera_color_data)
                bottom_camera_info_data = deepcopy(self.bottom_camera_info_data)
                bottom_camera_depth_data = deepcopy(self.bottom_camera_depth_data)

            points_top_image = self.detect_ar_tags_corners(top_camera_color_data)  
            points_top_world = []
            for point_top_image in points_top_image:
                validity, point_top_world = self.pixel2World(top_camera_info_data, int(point_top_image[0]), int(point_top_image[1]), top_camera_depth_data)
                if not validity:
                    print("invalid")
                    lol = input()
                points_top_world.append(point_top_world)
            points_top_world = np.array(points_top_world)

            points_bottom_image = self.detect_ar_tags_corners(bottom_camera_color_data)  
            points_bottom_world = []
            for point_bottom_image in points_bottom_image:
                validity, point_bottom_world = self.pixel2World(bottom_camera_info_data, int(point_bottom_image[0]), int(point_bottom_image[1]), bottom_camera_depth_data)
                if not validity:
                    print("invalid")
                    lol = input()
                points_bottom_world.append(point_bottom_world)
            points_bottom_world = np.array(points_bottom_world)

            print("points_top_world.shape: ", points_top_world.shape)
            print("points_bottom_world.shape: ", points_bottom_world.shape)

            rmsd, rotation, translation = self.kabschUmeyamaNoScale(points_top_world, points_bottom_world)

            self.visualizeTopVoxels(points_top_world)

            # points_bottom_world = translation.reshape(3,1) + rotation @ points_bottom_world[:,:,np.newaxis]
            self.visualizeBottomVoxels(points_bottom_world)

            trans = np.zeros((4,4))
            trans[:3,:3] = rotation
            trans[:3,3] = translation
            trans[3,3] = 1

            trans = top_cc @ trans @ np.linalg.inv(bottom_cc)

            print("Transformation Matrix from AR Tag Calibration: \n", trans)
            print("Launch file format: ",np.concatenate((trans[:3,3], Rotation.from_matrix(trans[:3,:3]).as_quat())).tolist())

            tf_trans = self.get_transformation_from_tf()
            print("Transformation Matrix from TF: \n", tf_trans)
            print("Launch file format: ",np.concatenate((tf_trans[:3,3], Rotation.from_matrix(tf_trans[:3,:3]).as_quat())).tolist())

            if rmsd < self.min_rmsd:
                self.min_rmsd = rmsd
                self.found_transform = trans

            # self.found_transforms.append(np.concatenate((translation, Rotation.from_matrix(rotation).as_quat())).tolist())

            # cv2.imshow('visualize_image123', viz_image)
            # cv2.waitKey(10)
            rate.sleep()

    def signal_handler(self, signal, frame):
        print("self.min_rmsd: ",self.min_rmsd)
        print("self.found_transform: ",self.found_transform)
        print("Launch file format: ",np.concatenate((self.found_transform[:3,3], Rotation.from_matrix(self.found_transform[:3,:3]).as_quat())).tolist())
        # for transform in self.found_transforms:
        #     print(transform)
        print("\nprogram exiting gracefully")
        sys.exit(0)   
        
if __name__ == '__main__':

    dual_camera_calibration = DualCameraCalibration()
    signal.signal(signal.SIGINT, dual_camera_calibration.signal_handler)
    time.sleep(1)
    dual_camera_calibration.get_transformation_from_ar_tags()

    rospy.spin()