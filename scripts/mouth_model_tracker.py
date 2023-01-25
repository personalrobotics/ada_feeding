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
import threading

lock = threading.Lock()
state = 0
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

        rate = rospy.Rate(1000.0)

        print("Press anything to start movement infront of mouth: ")
        lol = input()   

        previous_state = state
        current_state = state

        first_frame_captured = False

        while True: 

            rate.sleep()
            
            with lock:
                current_state = state

            if current_state != previous_state:
                print("Switching to state: ",current_state)
                first_frame_captured = False
                previous_state = current_state

            print("Current state: ",current_state)

            if current_state == 0: # move to infront of mouth
                
                # trajectory positions
                if not first_frame_captured:

                    while not rospy.is_shutdown():
                        try:
                            # print("Looking for transform")
                            transform_target = tfBuffer.lookup_transform('base_link', "forque_target", rospy.Time())
                            break
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                            rate.sleep()
                            continue

                    forque_target_base = np.zeros((4,4))
                    forque_target_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                    forque_target_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                    forque_target_base[3,3] = 1

                    first_frame_captured = True

                    servo_point_forque_target = np.zeros((4,4))
                    servo_point_forque_target[0, 0] = 1
                    servo_point_forque_target[1, 1] = 1
                    servo_point_forque_target[2, 2] = 1
                    servo_point_forque_target[:3,3] = np.array([0, 0, -0.08]).reshape(1,3)
                    servo_point_forque_target[3,3] = 1

                    servo_point_base = forque_target_base @ servo_point_forque_target

                # t = TransformStamped()

                # t.header.stamp = rospy.Time.now()
                # t.header.frame_id = "base_link"
                # t.child_frame_id = "servo_point"

                # t.transform.translation.x = servo_point_base[0][3]
                # t.transform.translation.y = servo_point_base[1][3]
                # t.transform.translation.z = servo_point_base[2][3]

                # R = Rotation.from_matrix(servo_point_base[:3,:3]).as_quat()
                # t.transform.rotation.x = R[0]
                # t.transform.rotation.y = R[1]
                # t.transform.rotation.z = R[2]
                # t.transform.rotation.w = R[3]

                # broadcaster.sendTransform(t)

                # current position
                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                key_times = [0, 1]
                key_rots = Rotation.concatenate((Rotation.from_matrix(servo_point_base[:3,:3]), Rotation.from_matrix(forque_base[:3,:3])))
                slerp = Slerp(key_times, key_rots)

                distance = np.linalg.norm(forque_base[:3,3] - servo_point_base[:3,3])
                resolution = 0.01
                
                position_increments = resolution*(forque_base[:3,3] - servo_point_base[:3,3]).reshape(1,3)/distance

                interp_positions = []
                for i in range(int(distance/resolution)+1):
                    interp_positions.append(servo_point_base[:3,3].reshape(1,3) + i*position_increments)
                interp_positions.append(forque_base[:3,3].reshape(1,3))
                interp_positions = np.array(interp_positions)

                interp_rotations = slerp(np.linspace(0.,1.,interp_positions.shape[0]))

                targets = []
                targets_positon = []

                for i in range(interp_positions.shape[0]):

                    interp_base = np.zeros((4,4))
                    interp_base[:3,:3] = interp_rotations[i].as_matrix()
                    interp_base[:3,3] = interp_positions[i]
                    interp_base[3,3] = 1

                    targets.append(interp_base)
                    targets_positon.append(np.array([interp_base[0][3], interp_base[1][3], interp_base[2][3]]))

                    # t = TransformStamped()

                    # t.header.stamp = rospy.Time.now()
                    # t.header.frame_id = "base_link"
                    # t.child_frame_id = "tcp_target" + str(i)

                    # t.transform.translation.x = interp_base[0][3]
                    # t.transform.translation.y = interp_base[1][3]
                    # t.transform.translation.z = interp_base[2][3]

                    # R = Rotation.from_matrix(interp_base[:3,:3]).as_quat()
                    # t.transform.rotation.x = R[0]
                    # t.transform.rotation.y = R[1]
                    # t.transform.rotation.z = R[2]
                    # t.transform.rotation.w = R[3]

                    # broadcaster.sendTransform(t)

                targets.reverse()
                targets_positon.reverse()

                targets_positon = np.array(targets_positon)

                current_position = np.array([forque_base[0][3], forque_base[1][3], forque_base[2][3]])

                diff = targets_positon - current_position
                dist = np.linalg.norm(diff, axis=1)
                
                distance_lookahead = 0.04

                tracking_index = dist.argmin()
                
                while tracking_index < len(targets) - 1:
                    if dist[tracking_index] > distance_lookahead:
                        break
                    tracking_index += 1

                print("distances: ",dist)
                print("tracking_index: ",tracking_index)    
                # print("targets_positon shape: ",targets_positon.shape)
                print("tracking target: ",targets[tracking_index][0][3], targets[tracking_index][1][3], targets[tracking_index][2][3])


                cartesian_point = CartesianTrajectoryPoint()
                
                goal = Pose()
                goal.position.x = targets[tracking_index][0][3]
                goal.position.y = targets[tracking_index][1][3]
                goal.position.z = targets[tracking_index][2][3]

                R = Rotation.from_matrix(targets[tracking_index][:3,:3]).as_quat()
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
                t.child_frame_id = "tcp_target"

                t.transform.translation.x = targets[tracking_index][0][3]
                t.transform.translation.y = targets[tracking_index][1][3]
                t.transform.translation.z = targets[tracking_index][2][3]

                R = Rotation.from_matrix(targets[tracking_index][:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

            elif current_state == 1: # visual servo

                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform_target = tfBuffer.lookup_transform('base_link', "forque_target", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_target_base = np.zeros((4,4))
                forque_target_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                forque_target_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                forque_target_base[3,3] = 1

                first_frame_captured = True

                servo_point_forque_target = np.zeros((4,4))
                servo_point_forque_target[0, 0] = 1
                servo_point_forque_target[1, 1] = 1
                servo_point_forque_target[2, 2] = 1
                servo_point_forque_target[:3,3] = np.array([0, 0, -0.08]).reshape(1,3)
                servo_point_forque_target[3,3] = 1

                servo_point_base = forque_target_base @ servo_point_forque_target

                cartesian_point = CartesianTrajectoryPoint()
                
                goal = Pose()
                goal.position.x = servo_point_base[0][3]
                goal.position.y = servo_point_base[1][3]
                goal.position.z = servo_point_base[2][3]

                R = Rotation.from_matrix(servo_point_base[:3,:3]).as_quat()
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
                t.child_frame_id = "tcp_target"

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

                if force_threshold_execeeded:
                    with lock:
                        state = 3
                    continue

                if not first_frame_captured:

                    while not rospy.is_shutdown():
                        try:
                            print("Looking for transform")
                            transform_target = tfBuffer.lookup_transform('base_link', "forque_target", rospy.Time())
                            break
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                            rate.sleep()
                            continue

                    forque_target_base = np.zeros((4,4))
                    forque_target_base[:3,:3] = Rotation.from_quat([transform_target.transform.rotation.x, transform_target.transform.rotation.y, transform_target.transform.rotation.z, transform_target.transform.rotation.w]).as_matrix()
                    forque_target_base[:3,3] = np.array([transform_target.transform.translation.x, transform_target.transform.translation.y, transform_target.transform.translation.z]).reshape(1,3)
                    forque_target_base[3,3] = 1

                    t = TransformStamped()

                    t.header.stamp = rospy.Time.now()
                    t.header.frame_id = "base_link"
                    t.child_frame_id = "inside_mouth_target"

                    t.transform.translation.x = forque_target_base[0][3]
                    t.transform.translation.y = forque_target_base[1][3]
                    t.transform.translation.z = forque_target_base[2][3]

                    R = Rotation.from_matrix(forque_target_base[:3,:3]).as_quat()
                    t.transform.rotation.x = R[0]
                    t.transform.rotation.y = R[1]
                    t.transform.rotation.z = R[2]
                    t.transform.rotation.w = R[3]

                    broadcaster.sendTransform(t)

                    # print("Input [1] to move inside: ")
                    # lol = input()
                    # if int(lol) == 1:
                    #     first_frame_captured = True
                    print("Updating forque target pose")

                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                if(np.linalg.norm(forque_base[:3,3] - forque_target_base[:3,3]) < 0.04):
                    first_frame_captured = True

                key_times = [0, 1]
                key_rots = Rotation.concatenate((Rotation.from_matrix(forque_target_base[:3,:3]), Rotation.from_matrix(forque_base[:3,:3])))
                slerp = Slerp(key_times, key_rots)

                distance = np.linalg.norm(forque_base[:3,3] - forque_target_base[:3,3])
                resolution = 0.005
                
                position_increments = resolution*(forque_base[:3,3] - forque_target_base[:3,3]).reshape(1,3)/distance

                interp_positions = []
                for i in range(int(distance/resolution)+1):
                    interp_positions.append(forque_target_base[:3,3].reshape(1,3) + i*position_increments)
                interp_positions.append(forque_base[:3,3].reshape(1,3))
                interp_positions = np.array(interp_positions)

                interp_rotations = slerp(np.linspace(0.,1.,interp_positions.shape[0]))

                targets = []
                targets_positon = []

                for i in range(interp_positions.shape[0]):

                    interp_base = np.zeros((4,4))
                    interp_base[:3,:3] = interp_rotations[i].as_matrix()
                    interp_base[:3,3] = interp_positions[i]
                    interp_base[3,3] = 1

                    targets.append(interp_base)
                    targets_positon.append(np.array([interp_base[0][3], interp_base[1][3], interp_base[2][3]]))

                    # t = TransformStamped()

                    # t.header.stamp = rospy.Time.now()
                    # t.header.frame_id = "base_link"
                    # t.child_frame_id = "tcp_target" + str(i)

                    # t.transform.translation.x = interp_base[0][3]
                    # t.transform.translation.y = interp_base[1][3]
                    # t.transform.translation.z = interp_base[2][3]

                    # R = Rotation.from_matrix(interp_base[:3,:3]).as_quat()
                    # t.transform.rotation.x = R[0]
                    # t.transform.rotation.y = R[1]
                    # t.transform.rotation.z = R[2]
                    # t.transform.rotation.w = R[3]

                    # broadcaster.sendTransform(t)

                targets.reverse()
                targets_positon.reverse()

                targets_positon = np.array(targets_positon)

                current_position = np.array([forque_base[0][3], forque_base[1][3], forque_base[2][3]])

                diff = targets_positon - current_position
                dist = np.linalg.norm(diff, axis=1)
                
                distance_lookahead = 0.02

                tracking_index = dist.argmin()
                
                while tracking_index < len(targets) - 1:
                    if dist[tracking_index] > distance_lookahead:
                        break
                    tracking_index += 1

                # print("distances: ",dist)
                print("tracking_index: ",tracking_index)    
                # print("targets_positon shape: ",targets_positon.shape)
                print("tracking target: ",targets[tracking_index][0][3], targets[tracking_index][1][3], targets[tracking_index][2][3])


                cartesian_point = CartesianTrajectoryPoint()
                
                goal = Pose()
                goal.position.x = targets[tracking_index][0][3]
                goal.position.y = targets[tracking_index][1][3]
                goal.position.z = targets[tracking_index][2][3]

                R = Rotation.from_matrix(targets[tracking_index][:3,:3]).as_quat()
                goal.orientation.x = R[0]
                goal.orientation.y = R[1]
                goal.orientation.z = R[2]
                goal.orientation.w = R[3]

                # print("Publishing goal: ",goal)
                cartesian_point.point.pose = goal
                cmd_pub.publish(cartesian_point)

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "tcp_target"

                t.transform.translation.x = targets[tracking_index][0][3]
                t.transform.translation.y = targets[tracking_index][1][3]
                t.transform.translation.z = targets[tracking_index][2][3]

                R = Rotation.from_matrix(targets[tracking_index][:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

            elif current_state == 3: # move outside mouth

                if not first_frame_captured:

                    while not rospy.is_shutdown():
                        try:
                            print("Looking for transform")
                            transform = tfBuffer.lookup_transform('base_link', "forque_target", rospy.Time())
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
                    forque_target_source[:3,3] = np.array([0, 0, -0.12]).reshape(1,3)
                    forque_target_source[3,3] = 1

                    forque_target_base = source_base @ forque_target_source

                    first_frame_captured = True

                while not rospy.is_shutdown():
                    try:
                        print("Looking for transform")
                        transform = tfBuffer.lookup_transform('base_link', "forque", rospy.Time())
                        break
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                        rate.sleep()
                        continue

                forque_base = np.zeros((4,4))
                forque_base[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
                forque_base[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
                forque_base[3,3] = 1

                key_times = [0, 1]
                key_rots = Rotation.concatenate((Rotation.from_matrix(forque_target_base[:3,:3]), Rotation.from_matrix(forque_base[:3,:3])))
                slerp = Slerp(key_times, key_rots)

                distance = np.linalg.norm(forque_base[:3,3] - forque_target_base[:3,3])
                resolution = 0.005
                
                position_increments = resolution*(forque_base[:3,3] - forque_target_base[:3,3]).reshape(1,3)/distance

                interp_positions = []
                for i in range(int(distance/resolution)+1):
                    interp_positions.append(forque_target_base[:3,3].reshape(1,3) + i*position_increments)
                interp_positions.append(forque_base[:3,3].reshape(1,3))
                interp_positions = np.array(interp_positions)

                interp_rotations = slerp(np.linspace(0.,1.,interp_positions.shape[0]))

                targets = []
                targets_positon = []

                for i in range(interp_positions.shape[0]):

                    interp_base = np.zeros((4,4))
                    interp_base[:3,:3] = interp_rotations[i].as_matrix()
                    interp_base[:3,3] = interp_positions[i]
                    interp_base[3,3] = 1

                    targets.append(interp_base)
                    targets_positon.append(np.array([interp_base[0][3], interp_base[1][3], interp_base[2][3]]))

                    # t = TransformStamped()

                    # t.header.stamp = rospy.Time.now()
                    # t.header.frame_id = "base_link"
                    # t.child_frame_id = "tcp_target" + str(i)

                    # t.transform.translation.x = interp_base[0][3]
                    # t.transform.translation.y = interp_base[1][3]
                    # t.transform.translation.z = interp_base[2][3]

                    # R = Rotation.from_matrix(interp_base[:3,:3]).as_quat()
                    # t.transform.rotation.x = R[0]
                    # t.transform.rotation.y = R[1]
                    # t.transform.rotation.z = R[2]
                    # t.transform.rotation.w = R[3]

                    # broadcaster.sendTransform(t)

                targets.reverse()
                targets_positon.reverse()

                targets_positon = np.array(targets_positon)

                current_position = np.array([forque_base[0][3], forque_base[1][3], forque_base[2][3]])

                diff = targets_positon - current_position
                dist = np.linalg.norm(diff, axis=1)
                
                distance_lookahead = 0.02

                tracking_index = dist.argmin()
                
                while tracking_index < len(targets) - 1:
                    if dist[tracking_index] > distance_lookahead:
                        break
                    tracking_index += 1

                print("distances: ",dist)
                print("tracking_index: ",tracking_index)    
                # print("targets_positon shape: ",targets_positon.shape)
                print("tracking target: ",targets[tracking_index][0][3], targets[tracking_index][1][3], targets[tracking_index][2][3])


                cartesian_point = CartesianTrajectoryPoint()
                
                goal = Pose()
                goal.position.x = targets[tracking_index][0][3]
                goal.position.y = targets[tracking_index][1][3]
                goal.position.z = targets[tracking_index][2][3]

                R = Rotation.from_matrix(targets[tracking_index][:3,:3]).as_quat()
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
                t.child_frame_id = "tcp_target"

                t.transform.translation.x = targets[tracking_index][0][3]
                t.transform.translation.y = targets[tracking_index][1][3]
                t.transform.translation.z = targets[tracking_index][2][3]

                R = Rotation.from_matrix(targets[tracking_index][:3,:3]).as_quat()
                t.transform.rotation.x = R[0]
                t.transform.rotation.y = R[1]
                t.transform.rotation.z = R[2]
                t.transform.rotation.w = R[3]

                broadcaster.sendTransform(t)

    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)