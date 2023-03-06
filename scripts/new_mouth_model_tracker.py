#! /usr/bin/env python

import sys
import rospy
import tf2_ros
import actionlib
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from geometry_msgs.msg import TransformStamped, WrenchStamped
from moveit_msgs.msg import CartesianTrajectoryPoint
from std_msgs.msg import Int64
from std_msgs.msg import String
import threading
import time

# Parameters
OPEN_LOOP_RADIUS = 0.015
INTERMEDIATE_THRESHOLD = 0.014
DISTANCE_LOOKAHEAD = 0.02
ANGULAR_LOOKAHEAD = 5*np.pi/180
DISTANCE_INFRONT_MOUTH = 0.10
MOVE_OUTSIDE_DISTANCE = 0.14

lock = threading.Lock()
state = 1
force_threshold_execeeded = False 

def callback(msg):

    global state
    with lock:
        state = msg.data

def ft_callback(msg):

    global force_threshold_execeeded
    # print("Force Magnitude: ",np.linalg.norm(np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])))
    # if np.linalg.norm(np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])) > 3:
    #     force_threshold_execeeded = True
        # print("True")

def get_angular_distance(rotation_a, rotation_b):
    return np.linalg.norm(Rotation.from_matrix(np.dot(rotation_a, rotation_b.T)).as_rotvec())


def get_next_waypoint(source, target, distance_lookahead = DISTANCE_LOOKAHEAD, angular_lookahead = ANGULAR_LOOKAHEAD):

    position_error = np.linalg.norm(source[:3,3] - target[:3,3])
    orientation_error = get_angular_distance(source[:3,:3], target[:3,:3])


    next_waypoint = np.zeros((4,4))
    next_waypoint[3,3] = 1

    if position_error <= distance_lookahead:
        next_waypoint[:3,3] = target[:3,3]
    else:    
        next_waypoint[:3,3] = source[:3,3].reshape(1,3) + distance_lookahead*(target[:3,3] - source[:3,3]).reshape(1,3)/position_error

    if orientation_error <= angular_lookahead:
        next_waypoint[:3,:3] = target[:3,:3]
    else:
        key_times = [0, 1]
        key_rots = Rotation.concatenate((Rotation.from_matrix(source[:3,:3]), Rotation.from_matrix(target[:3,:3])))
        slerp = Slerp(key_times, key_rots)

        interp_rotations = slerp([angular_lookahead/orientation_error]) #second last is also aligned
        next_waypoint[:3,:3] = interp_rotations[0].as_matrix()

    return next_waypoint


if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('jointgroup_test_py')
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        broadcaster = tf2_ros.TransformBroadcaster()

        move_inside_sub = rospy.Subscriber('/move_inside', Int64, callback)
        ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, ft_callback)

        cmd_pub = rospy.Publisher('/task_space_compliant_controller/command', CartesianTrajectoryPoint, queue_size=10)
        mode_cmd_pub = rospy.Publisher('/task_space_compliant_controller/mode', String, queue_size=10)

        rate = rospy.Rate(60.0)
        slow_rate = rospy.Rate(2.0)

        # print("Press anything to start maintain contact:")
        # lol = input()

        # mode_command = String()
        # mode_command.data = "zero_contact"
        # mode_cmd_pub.publish(mode_command)

        print("Press anything to start movement infront of mouth: ")
        lol = input()   

        closed_loop = True

        previous_state = state
        current_state = state

        last_time = time.time()
        while True: 

            rate.sleep()

            print("Frequency: ",1.0/(time.time() - last_time))
            last_time = time.time()
            
            with lock:
                current_state = state

            if current_state != previous_state:
                print("Switching to state: ",current_state)
                closed_loop = True
                previous_state = current_state

            print("Current state: ",current_state)

            if current_state == 1: # move to infront of mouth
                
                # trajectory positions
                if closed_loop:

                    # while not rospy.is_shutdown():
                    #     try:
                    #         # print("Looking for transform")
                    #         transform_target = tfBuffer.lookup_transform('base_link', "forque_end_effector_target", rospy.Time())
                    #         break
                    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    #         rate.sleep()
                    #         continue

                    # # print("Transform: ", transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w)

                    # forque_target_base = np.zeros((4,4))
                    # forque_target_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                    # forque_target_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                    # forque_target_base[3,3] = 1


                    forque_target_base = np.array([[-0.90093711, -0.1803962, 0.39467648, 0.42479337], 
                        [ 0.41205567, -0.07038973,  0.90843569,  0.16312421],
                        [-0.13609718, 0.98107212, 0.13775, 0.69508812],
                        [ 0., 0., 0., 1.]])


                    closed_loop = False

                    servo_point_forque_target = np.zeros((4,4))
                    servo_point_forque_target[0, 0] = 1
                    servo_point_forque_target[1, 1] = 1
                    servo_point_forque_target[2, 2] = 1
                    servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)
                    servo_point_forque_target[3,3] = 1

                    servo_point_base = forque_target_base @ servo_point_forque_target

                    mode_command = String()
                    mode_command.data = "use_pose_integral"
                    # mode_command.data = "zero_contact"
                    # mode_command.data = "none"
                    mode_cmd_pub.publish(mode_command)
                    time.sleep(0.5)

                    mode_command = String()
                    # mode_command.data = "use_pose_integral"
                    # mode_command.data = "zero_contact"
                    mode_command.data = "low_stiffness"
                    mode_cmd_pub.publish(mode_command)
                    time.sleep(0.1)

                # current position
                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                target = get_next_waypoint(forque_base, servo_point_base)
                # target = servo_point_base

                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                # print("Publishing goal: ",goal)
                cartesian_point = CartesianTrajectoryPoint()
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "next_target"

                t.transform.translation.x = target[0][3]
                t.transform.translation.y = target[1][3]
                t.transform.translation.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "final_target"

                t.transform.translation.x = servo_point_base[0][3]
                t.transform.translation.y = servo_point_base[1][3]
                t.transform.translation.z = servo_point_base[2][3]

                R = Rotation.from_matrix(servo_point_base[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

            elif current_state == 2: # move inside mouth

                if closed_loop:

                    mode_command = String()
                    mode_command.data = "use_pose_integral"
                    # mode_command.data = "zero_contact"
                    # mode_command.data = "low_stiffness"
                    mode_cmd_pub.publish(mode_command)
                    time.sleep(0.1)

                    # while not rospy.is_shutdown():
                    #     try:
                    #         print("Looking for transform")
                    #         transform_target = tfBuffer.lookup_transform('base_link', "forque_end_effector_target", rospy.Time())
                    #         break
                    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    #         rate.sleep()
                    #         continue

                    # forque_target_base = np.zeros((4,4))
                    # forque_target_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                    # forque_target_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                    # forque_target_base[3,3] = 1

                    # transfer_forque_target = np.zeros((4,4))
                    # transfer_forque_target[0, 0] = 1
                    # transfer_forque_target[1, 1] = 1
                    # transfer_forque_target[2, 2] = 1
                    # transfer_forque_target[:3,3] = np.array([0, 0, 0.03]).reshape(1,3)
                    # transfer_forque_target[3,3] = 1

                    # forque_target_base = forque_target_base @ transfer_forque_target

                    forque_target_base = np.array([[-0.90093711, -0.1803962, 0.39467648, 0.42479337], 
                        [ 0.41205567, -0.07038973,  0.90843569,  0.16312421],
                        [-0.13609718, 0.98107212, 0.13775, 0.69508812],
                        [ 0., 0., 0., 1.]])

                    closed_loop = False

                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_source = np.zeros((4,4))
                forque_source[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_source[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_source[3,3] = 1

                distance = np.linalg.norm(forque_source[:3,3] - forque_target_base[:3,3])

                if distance < OPEN_LOOP_RADIUS:
                    closed_loop = False

                intermediate_forque_target = np.zeros((4,4))
                intermediate_forque_target[0, 0] = 1
                intermediate_forque_target[1, 1] = 1
                intermediate_forque_target[2, 2] = 1
                intermediate_forque_target[:3,3] = np.array([0, 0, -distance]).reshape(1,3)
                intermediate_forque_target[3,3] = 1

                intermediate_forque_target = forque_target_base @ intermediate_forque_target

                intermediate_position_error = np.linalg.norm(forque_source[:3,3] - intermediate_forque_target[:3,3])
                intermediate_angular_error = get_angular_distance(forque_source[:3,:3], intermediate_forque_target[:3,:3])

                print("intermediate_position_error: ", forque_source[:3,3] - intermediate_forque_target[:3,3])
                print("intermediate_position_error mag: ", intermediate_position_error)
                print("intermediate_angular_error: ", intermediate_angular_error)

                ipe_forque_frame = np.linalg.inv(forque_source[:3,:3]) @ (forque_source[:3,3] - intermediate_forque_target[:3,3]).reshape(3,1)
                print("Error in forque frame: ",ipe_forque_frame)
                error_mag = np.linalg.norm(np.array([ipe_forque_frame[0], ipe_forque_frame[1]]))
                print("Error mag:",error_mag)
 
                if intermediate_position_error > INTERMEDIATE_THRESHOLD: # The thresholds here should be ideally larger than the thresholds for tracking trajectories
                    print("Tracking intermediate position... ")
                    target = get_next_waypoint(forque_source, intermediate_forque_target)
                else:
                    print("Tracking target position... ")
                    target = get_next_waypoint(intermediate_forque_target, forque_target_base)
                
                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                # print("Publishing goal: ",goal)
                cartesian_point = CartesianTrajectoryPoint()
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "next_target"

                t.transform.translation.x = target[0][3]
                t.transform.translation.y = target[1][3]
                t.transform.translation.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "final_target"

                t.transform.translation.x = forque_target_base[0][3]
                t.transform.translation.y = forque_target_base[1][3]
                t.transform.translation.z = forque_target_base[2][3]

                R = Rotation.from_matrix(forque_target_base[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "intermediate_target"

                t.transform.translation.x = intermediate_forque_target[0][3]
                t.transform.translation.y = intermediate_forque_target[1][3]
                t.transform.translation.z = intermediate_forque_target[2][3]

                R = Rotation.from_matrix(intermediate_forque_target[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

            elif current_state == 3: # move outside mouth

                if closed_loop:

                    mode_command = String()
                    # mode_command.data = "use_pose_integral"
                    # mode_command.data = "zero_contact"
                    mode_command.data = "high_stiffness"
                    mode_cmd_pub.publish(mode_command)
                    time.sleep(0.1)

                    while not rospy.is_shutdown():
                        try:
                            print("Looking for transform")
                            transform = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                            break
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                            rate.sleep()
                            continue

                    source_base = np.zeros((4,4))
                    source_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                    source_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                    source_base[3,3] = 1

                    forque_target_source = np.zeros((4,4))
                    forque_target_source[0, 0] = 1
                    forque_target_source[1, 1] = 1
                    forque_target_source[2, 2] = 1
                    forque_target_source[:3,3] = np.array([0, 0, -MOVE_OUTSIDE_DISTANCE]).reshape(1,3)
                    forque_target_source[3,3] = 1

                    forque_target_base = source_base @ forque_target_source

                    closed_loop = False

                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                target = get_next_waypoint(forque_base, forque_target_base)
                
                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                # print("Publishing goal: ",goal)
                cartesian_point = CartesianTrajectoryPoint()
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "next_target"

                t.transform.translation.x = target[0][3]
                t.transform.translation.y = target[1][3]
                t.transform.translation.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "final_target"

                t.transform.translation.x = forque_target_base[0][3]
                t.transform.translation.y = forque_target_base[1][3]
                t.transform.translation.z = forque_target_base[2][3]

                R = Rotation.from_matrix(forque_target_base[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

            elif current_state == 4: # tilt inside mouth

                # trajectory positions
                if closed_loop:

                    while not rospy.is_shutdown():
                        try:
                            # print("Looking for transform")
                            transform_target = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                            break
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                            rate.sleep()
                            continue

                    forque_base = np.zeros((4,4))
                    forque_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                    forque_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                    forque_base[3,3] = 1

                    closed_loop = False

                    forque_target_base = np.zeros((4,4))
                    forque_target_base[:3,:3] = Rotation.from_euler('X', 15*np.pi/180).as_matrix()
                    forque_target_base[3,3] = 1

                    forque_target_base = forque_base @ forque_target_base

                    # mode_command = String()
                    # mode_command.data = "zero_contact"
                    # mode_cmd_pub.publish(mode_command)
                    # time.sleep(0.5)

                while not rospy.is_shutdown():
                    try:
                        # print("Looking for transform")
                        transform_target = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                target = get_next_waypoint(forque_base, forque_target_base)

                cartesian_point = CartesianTrajectoryPoint()
                
                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                print("Publishing goal: ",goal)
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "next_target"

                t.transform.translation.x = target[0][3]
                t.transform.translation.y = target[1][3]
                t.transform.translation.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

            elif current_state == 5: # tilt inside mouth and move outside

                # trajectory positions
                if closed_loop:

                    while not rospy.is_shutdown():
                        try:
                            # print("Looking for transform")
                            transform_target = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                            break
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                            rate.sleep()
                            continue

                    forque_base = np.zeros((4,4))
                    forque_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                    forque_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                    forque_base[3,3] = 1

                    closed_loop = False

                    planar_transform = np.zeros((4,4))
                    planar_transform[:3,:3] = Rotation.from_euler('X', -15*np.pi/180).as_matrix()
                    planar_transform[3,3] = 1

                    forque_target_base = forque_base @ planar_transform

                    back_translation = np.zeros((4,4))
                    back_translation[0, 0] = 1
                    back_translation[1, 1] = 1
                    back_translation[2, 2] = 1
                    back_translation[:3,3] = np.array([0, 0, -0.1]).reshape(1,3)
                    back_translation[3,3] = 1

                    forque_target_base = forque_target_base @ back_translation
                    forque_target_base = forque_target_base @ np.linalg.inv(planar_transform)

                    # mode_command = String()
                    # mode_command.data = "maintain_contact"
                    # mode_cmd_pub.publish(mode_command)
                    # time.sleep(1)

                while not rospy.is_shutdown():
                    try:
                        # print("Looking for transform")
                        transform_target = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                target = get_next_waypoint(forque_base, forque_target_base)

                cartesian_point = CartesianTrajectoryPoint()
                
                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                print("Publishing goal: ",goal)
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "next_target"

                t.transform.translation.x = target[0][3]
                t.transform.translation.y = target[1][3]
                t.transform.translation.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

            elif current_state == 6: # move to infront of mouth

                if closed_loop:
                    print("Publihsing zero")
                    mode_command = String()
                    # mode_command.data = "use_pose_integral"
                    mode_command.data = "zero_contact"
                    # mode_command.data = "none"
                    mode_cmd_pub.publish(mode_command)
                    time.sleep(0.5)
                    closed_loop = False

            elif current_state == 7: # move to infront of mouth

                if closed_loop:
                    mode_command = String()
                    # mode_command.data = "use_pose_integral"
                    # mode_command.data = "zero_contact"
                    mode_command.data = "weird_compliance"
                    mode_cmd_pub.publish(mode_command)
                    time.sleep(0.5)
                    closed_loop = False

            elif current_state == 8: # move to infront of mouth

                # current position
                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                target = forque_base

                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                # print("Publishing goal: ",goal)
                cartesian_point = CartesianTrajectoryPoint()
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "next_target"

                t.transform.translation.x = target[0][3]
                t.transform.translation.y = target[1][3]
                t.transform.translation.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

                slow_rate.sleep()

            elif current_state == 9: # move to infront of mouth

                if closed_loop:
                    mode_command = String()
                    # mode_command.data = "use_pose_integral"
                    # mode_command.data = "zero_contact"
                    mode_command.data = "high_stiffness"
                    mode_cmd_pub.publish(mode_command)
                    time.sleep(0.5)
                    closed_loop = False


            elif current_state == 10: # move to infront of mouth
                
                # trajectory positions
                if closed_loop:

                    forque_target_base = np.array([[-0.90093711, -0.1803962, 0.39467648, 0.42479337], 
                        [ 0.41205567, -0.07038973,  0.90843569,  0.16312421],
                        [-0.13609718, 0.98107212, 0.13775, 0.69508812],
                        [ 0., 0., 0., 1.]])

                    closed_loop = False

                    servo_point_forque_target = np.zeros((4,4))
                    servo_point_forque_target[0, 0] = 1
                    servo_point_forque_target[1, 1] = 1
                    servo_point_forque_target[2, 2] = 1
                    servo_point_forque_target[:3,3] = np.array([0, 0, -0.2]).reshape(1,3)
                    servo_point_forque_target[3,3] = 1

                    servo_point_base = forque_target_base @ servo_point_forque_target

                # current position
                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                # target = get_next_waypoint(forque_base, servo_point_base)
                target = servo_point_base

                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                # print("Publishing goal: ",goal)
                cartesian_point = CartesianTrajectoryPoint()
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

            elif current_state == 11: # move to infront of mouth
                
                # trajectory positions
                if closed_loop:

                    forque_target_base = np.array([[-0.90093711, -0.1803962, 0.39467648, 0.42479337], 
                        [ 0.41205567, -0.07038973,  0.90843569,  0.16312421],
                        [-0.13609718, 0.98107212, 0.13775, 0.69508812],
                        [ 0., 0., 0., 1.]])

                    closed_loop = False

                    servo_point_forque_target = np.zeros((4,4))
                    servo_point_forque_target[0, 0] = 1
                    servo_point_forque_target[1, 1] = 1
                    servo_point_forque_target[2, 2] = 1
                    servo_point_forque_target[:3,3] = np.array([0, 0, 0.0]).reshape(1,3)
                    servo_point_forque_target[3,3] = 1

                    servo_point_base = forque_target_base @ servo_point_forque_target

                # current position
                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque_end_effector", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                # target = get_next_waypoint(forque_base, servo_point_base)
                target = servo_point_base

                goal = Pose()
                goal.position.x = target[0][3]
                goal.position.y = target[1][3]
                goal.position.z = target[2][3]

                R = Rotation.from_matrix(target[:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                # print("Publishing goal: ",goal)
                cartesian_point = CartesianTrajectoryPoint()
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)