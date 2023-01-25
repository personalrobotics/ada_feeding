import os, sys
import cv2
import numpy as np
import time
import math

from scipy.spatial.transform import Rotation

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

class FoodBoundingBoxPerception:
    def __init__(self):
        rospy.init_node('FoodBoundingBoxPerception')

        self.top_camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)
        self.top_color_image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.top_depth_image_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

        self.bottom_camera_info_sub = message_filters.Subscriber("/bottom_camera/color/camera_info", CameraInfo)
        self.bottom_color_image_sub = message_filters.Subscriber("/bottom_camera/color/image_raw", Image)
        self.bottom_depth_image_sub = message_filters.Subscriber("/bottom_camera/aligned_depth_to_color/image_raw", Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.top_camera_info_sub, self.top_color_image_sub, self.top_depth_image_sub, self.bottom_camera_info_sub, self.bottom_color_image_sub, self.bottom_depth_image_sub], 100, 0.1)
        ts.registerCallback(self.multipleCameraCallback)

        self.bridge = CvBridge()

        self.cube_publisher =  rospy.Publisher("/food_perception/food_bounding_box/marker_array", MarkerArray, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.broadcaster = tf2_ros.TransformBroadcaster()
        print("initialized")

    # Any reason that the callback shouldn't be computation intensive?
    def multipleCameraCallback(self, top_camera_info_msg, top_color_image_msg, top_depth_image_msg, bottom_camera_info_msg, bottom_color_image_msg, bottom_depth_image_msg):

        # global viz_image

        print("In callback")
        try:
            # Convert your ROS Image message to OpenCV2
            top_color_image = self.bridge.imgmsg_to_cv2(top_color_image_msg, "bgr8")
            bottom_color_image = self.bridge.imgmsg_to_cv2(bottom_color_image_msg, "bgr8")

            top_depth_image = self.bridge.imgmsg_to_cv2(top_depth_image_msg, "32FC1")
            bottom_depth_image = self.bridge.imgmsg_to_cv2(bottom_depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        rate = rospy.Rate(1000.0)
        while not rospy.is_shutdown():
            try:
                print("Looking for transform 0")
                transform = self.tfBuffer.lookup_transform('base_link', 'camera_color_optical_frame', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue

        base_to_camera = np.zeros((4,4))
        base_to_camera[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        base_to_camera[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        base_to_camera[3,3] = 1

        camera_to_base = np.linalg.inv(base_to_camera)

        while not rospy.is_shutdown():
            try:
                print("Looking for transform 1")
                transform = self.tfBuffer.lookup_transform('base_link', 'bottom_camera_color_optical_frame', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue

        base_to_bottom_camera = np.zeros((4,4))
        base_to_bottom_camera[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        base_to_bottom_camera[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        base_to_bottom_camera[3,3] = 1      

        while not rospy.is_shutdown():
            try:
                print("Looking for transform 2")
                transform = self.tfBuffer.lookup_transform('camera_color_optical_frame', 'forque_end_effector', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue  

        #  forque tip in frame of reference of top camera
        top_camera_forque_tip_3D = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        top_camera_forque_tip_image = [561, 413]

        # print("top_camera_forque_tip_image: ", top_camera_forque_tip_image)

        # viz_image = top_color_image.copy()
        # cv2.circle(viz_image, (int(top_camera_forque_tip_image[0]), int(top_camera_forque_tip_image[1])), 4, (0, 0, 255), -1)
        # print("(shape): ", viz_image.shape)

        while not rospy.is_shutdown():
            try:
                print("Looking for transform 3")
                transform = self.tfBuffer.lookup_transform('bottom_camera_color_optical_frame', 'forque_end_effector', rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue  

        #  forque tip in frame of reference of bottom camera
        bottom_camera_forque_tip_3D = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        bottom_camera_forque_tip_image = [398, 85]

        print("bottom_camera_forque_tip_image: ", bottom_camera_forque_tip_image)

        bounding_box_half = 100

        depth_range = [0.25, 0.29] #26cm

        # max_y = 0
        # min_x = 0
        # max_x = 0
        # min_y = 0

        color = []

        top_food_voxels = []

        for i in range(-bounding_box_half,bounding_box_half+1):
            for j in range(-bounding_box_half,1):
                validity, world_pixel = self.pixel2World(top_camera_info_msg, top_camera_forque_tip_image[0]+i,  top_camera_forque_tip_image[1]+j, top_depth_image)
                if validity and world_pixel[2] > depth_range[0] and world_pixel[2] < depth_range[1] :
                    top_food_voxels.append(world_pixel)
                    color.append(top_color_image[top_camera_forque_tip_image[1]+j][top_camera_forque_tip_image[0]+i])

        bottom_food_voxels = []

        for i in range(-bounding_box_half,bounding_box_half+1):
            for j in range(30,bounding_box_half+1):
                validity, world_pixel = self.pixel2World(bottom_camera_info_msg, bottom_camera_forque_tip_image[0]+i, bottom_camera_forque_tip_image[1]+j, bottom_depth_image)
                if validity and world_pixel[2] > depth_range[0] and world_pixel[2] < depth_range[1] :
                    bottom_food_voxels.append(world_pixel)
                    color.append(top_color_image[bottom_camera_forque_tip_image[1]+j][bottom_camera_forque_tip_image[0]+i])

        top_food_voxels = np.array(top_food_voxels)
        bottom_food_voxels = np.array(bottom_food_voxels)
        color = np.array(color)

        print("top food_voxels shape: ", top_food_voxels.shape)
        print("bottom food voxels shape: ", bottom_food_voxels.shape)

        # top_food_voxels = np.squeeze(base_to_camera[:3,3].reshape(3,1) + base_to_camera[:3,:3] @ top_food_voxels[:,:,np.newaxis])

        bottom_food_voxels = np.squeeze(base_to_bottom_camera[:3,3].reshape(3,1) + base_to_bottom_camera[:3,:3] @ bottom_food_voxels[:,:,np.newaxis])
        bottom_food_voxels = np.squeeze(camera_to_base[:3,3].reshape(3,1) + camera_to_base[:3,:3] @ bottom_food_voxels[:,:,np.newaxis])

        food_voxels = np.concatenate((top_food_voxels, bottom_food_voxels), axis = 0)
        # food_voxels = bottom_food_voxels

        # Check the frame of top camera
        bounding_box_bottom_left_back = np.array([np.min(food_voxels[:,0]), np.min(food_voxels[:,1]), np.min(food_voxels[:,2])])
        bounding_box_top_right_ahead = np.array([np.max(food_voxels[:,0]), np.max(food_voxels[:,1]), np.max(food_voxels[:,2])])

        bounding_box_center = (bounding_box_bottom_left_back + bounding_box_top_right_ahead)/2.0
        bounding_box_dimensions = (bounding_box_top_right_ahead - bounding_box_bottom_left_back)

        color = np.mean(color, axis=0)/255.0

        print("bounding_box_center: ", bounding_box_center)
        print("bounding_box_dimensions: ", bounding_box_dimensions)
        print("color: ",color)

        self.visualizeCube(bounding_box_center, bounding_box_dimensions, color)

    def visualizeCube(self, bounding_box_center, bounding_box_dimensions, color):

        markerArray = MarkerArray()

        cube_marker = Marker()
        cube_marker.header.seq = 0
        cube_marker.header.stamp = rospy.Time.now()

        cube_marker.header.frame_id = "camera_color_optical_frame"  

        cube_marker.ns = "cube_marker"
        cube_marker.id =  1
        cube_marker.type = cube_marker.CUBE # CUBE LIST
        cube_marker.action = cube_marker.ADD # ADD
        cube_marker.lifetime = rospy.Duration()

        cube_marker.color.a = 1.0
        cube_marker.color.b = color[0]
        cube_marker.color.g = color[1]
        cube_marker.color.r = color[2]
        
        cube_marker.pose.position.x = bounding_box_center[0]
        cube_marker.pose.position.y = bounding_box_center[1]
        cube_marker.pose.position.z = bounding_box_center[2]

        cube_marker.scale.x = bounding_box_dimensions[0]
        cube_marker.scale.y = bounding_box_dimensions[1]
        cube_marker.scale.z = bounding_box_dimensions[2]

        markerArray.markers.append(cube_marker)

        self.cube_publisher.publish(markerArray)

    def pixel2World(self, camera_info, image_x, image_y, depth_image):

        # print("(image_y,image_x): ",image_y,image_x)
        # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

        if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
            # print("Out of bounds... ")
            return False, None

        depth = depth_image[image_y, image_x]

        if math.isnan(depth) or depth < 0.05 or depth > 1.0:

            depth = []
            for i in range(-2,2):
                for j in range(-2,2):
                    if image_y+i >= depth_image.shape[0] or image_x+j >= depth_image.shape[1]:
                        # print("Out of bounds... ")
                        return False, None
                    pixel_depth = depth_image[image_y+i, image_x+j]
                    if not (math.isnan(pixel_depth) or pixel_depth < 50 or pixel_depth > 1000):
                        depth += [pixel_depth]

            if len(depth) == 0:
                # print("No valid pixel :(")
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

        return True, (world_x, world_y, world_z)

    def world2Pixel(self, camera_info, world_voxel):

        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]  

        return fx * world_voxel[0] / world_voxel[2] + cx, fy * world_voxel[1] / world_voxel[2] + cy

if __name__ == '__main__':

    food_bounding_box_perception = FoodBoundingBoxPerception()

    # tfBuffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(tfBuffer)

    # while True:
    #     k1 = time.time()
    #     rate = rospy.Rate(10.0)
    #     while not rospy.is_shutdown():
    #         try:
    #             print("Looking for transform")
    #             transform = tfBuffer.lookup_transform('base_link', 'forque_end_effector', rospy.Time())
    #             break
    #         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #             continue
    #     rate.sleep()

    #     trans = np.zeros((4,4))
    #     trans[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
    #     trans[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
    #     trans[3,3] = 1

    #     head_perception.updateForqueTargetPose(trans)

    #     kk = np.repeat(np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3), 75, axis=0)

    #     print("KK Shape:", kk.shape)
    #     head_perception.visualizeVoxels(kk)

    #     k2 = time.time()
    #     print("Rate of publishing: ", 1.0/(k2 - k1))


    # time.sleep(2)
    # while True:
    #     # rospy.spinOnce()
    #     cv2.imshow('visualize_image123', viz_image)
    #     cv2.waitKey(10)

    rospy.spin()