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
from std_msgs.msg import String, Bool
import threading
import time

# Parameters
OPEN_LOOP_RADIUS = 0.01
# OPEN_LOOP_RADIUS = 0.0
INTERMEDIATE_THRESHOLD = 0.014
INTERMEDIATE_THRESHOLD_RELAXED = 0.02
INFRONT_DISTANCE_LOOKAHEAD = 0.04
INSIDE_DISTANCE_LOOKAHEAD_Z = 0.045
INSIDE_DISTANCE_LOOKAHEAD_XY = 0.025
TILT_DISTANCE_LOOKAHEAD = 0.05
TILT_ANGULAR_LOOKAHEAD = 10*np.pi/180
ANGULAR_LOOKAHEAD = 5*np.pi/180
DISTANCE_INFRONT_MOUTH = 0.10
MOVE_OUTSIDE_DISTANCE = 0.14
TILT_MOVE_OUTSIDE_DISTANCE = 0.14


class BiteTransferTrajectoryTracker:
    def __init__(self):

        rospy.init_node('jointgroup_test_py')
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.task_cmd_publisher = rospy.Publisher('/task_space_compliant_controller/command', CartesianTrajectoryPoint, queue_size=10)
        self.task_mode_publisher = rospy.Publisher('/task_space_compliant_controller/mode', String, queue_size=10)

        self.control_rate = rospy.Rate(100.0)

        self.state = 0
        self.state_lock = threading.Lock()

        self.mouth_target_pose = None

        self.force_threshold_execeeded = False
        self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ft_callback)

        self.mouth_open_detected = False
        self.mouth_state_sub = rospy.Subscriber('/head_perception/mouth_state', Bool, self.mouth_state_callback)

    def ft_callback(self, msg):

        # print("Force Magnitude: ",np.linalg.norm(np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])))
        if (not self.force_threshold_execeeded) and msg.wrench.force.y > 0.3:
            self.force_threshold_execeeded = True
            with self.state_lock:
                self.state = 3

    def mouth_state_callback(self, msg):

        if (not self.mouth_open_detected) and msg.data:
            self.mouth_open_detected = True
            with self.state_lock:
                self.state = 1

    def getAngularDistance(self, rotation_a, rotation_b):
        return np.linalg.norm(Rotation.from_matrix(np.dot(rotation_a, rotation_b.T)).as_rotvec())

    def getNextWaypoint(self, source, target, distance_lookahead, angular_lookahead = ANGULAR_LOOKAHEAD):

        position_error = np.linalg.norm(source[:3,3] - target[:3,3])
        orientation_error = self.getAngularDistance(source[:3,:3], target[:3,:3])

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

    def publishTaskMode(self, mode):
        mode_command = String()
        mode_command.data = mode
        self.task_mode_publisher.publish(mode_command)
        time.sleep(0.1)

    def publishTaskCommand(self, target):

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
        self.task_cmd_publisher.publish(cartesian_point)

    def publishTransformationToTF(self, source_frame, target_frame, transform):

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = transform[0][3]
        t.transform.translation.y = transform[1][3]
        t.transform.translation.z = transform[2][3]

        R = Rotation.from_matrix(transform[:3,:3]).as_quat()
        t.transform.rotation.x = R[0]
        t.transform.rotation.y = R[1]
        t.transform.rotation.z = R[2]
        t.transform.rotation.w = R[3]

        self.broadcaster.sendTransform(t)


    def getTransformationFromTF(self, source_frame, target_frame):

        while not rospy.is_shutdown():
            try:
                print("Looking for transform")
                transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.control_rate.sleep()
                continue

        T = np.zeros((4,4))
        T[:3,:3] = Rotation.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        T[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        T[3,3] = 1

        return T

    def runControlLoop(self):

        # print("Input starting state: ")
        # inp = input()
        # self.state = int(inp)
        closed_loop = True
        run_once = True

        # Assumption: No one will be updating state when this runs
        previous_state = self.state
        current_state = self.state

        last_time = time.time()
        while True: 

            self.control_rate.sleep()

            print("Frequency: ",1.0/(time.time() - last_time))
            last_time = time.time()
            
            with self.state_lock:
                current_state = self.state

            if current_state != previous_state:
                print("Switching to self.state: ",current_state)
                closed_loop = True
                run_once = True
                previous_state = current_state

            print("Current self.state: ",current_state)

            if current_state == 1: # move to infront of mouth
                
                # trajectory positions
                if closed_loop:

                    for i in range(0,10):

                        forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                        self.publishTaskCommand(forque_base)

                    time.sleep(0.1)

                    self.publishTaskMode("none")
                    self.publishTaskMode("default_stiffness")

                    forque_target_base = self.getTransformationFromTF("base_link", "forque_end_effector_target")

                    # forque_target_base = np.array([[-0.90093711, -0.1803962, 0.39467648, 0.42479337], 
                    #     [ 0.41205567, -0.07038973,  0.90843569,  0.16312421],
                    #     [-0.13609718, 0.98107212, 0.13775, 0.69508812],
                    #     [ 0., 0., 0., 1.]])

                    closed_loop = False

                    self.mouth_target_pose = forque_target_base

                    servo_point_forque_target = np.identity(4)
                    servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)

                    servo_point_base = forque_target_base @ servo_point_forque_target

                # current position
                forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")

                distance = np.linalg.norm(forque_base[:3,3] - servo_point_base[:3,3])

                if distance < INTERMEDIATE_THRESHOLD_RELAXED:
                    with self.state_lock:
                        self.state = 2

                target = self.getNextWaypoint(forque_base, servo_point_base, distance_lookahead=INFRONT_DISTANCE_LOOKAHEAD)

                self.publishTaskCommand(target)
                self.publishTransformationToTF("base_link", "next_target", target)
                self.publishTransformationToTF("base_link", "final_target", servo_point_base)

            elif current_state == 2: # move inside mouth

                if closed_loop:

                    if run_once:

                        for i in range(0,10):

                            forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                            self.publishTaskCommand(forque_base)

                        time.sleep(0.1)

                        self.publishTaskMode("use_pose_integral")
                        self.publishTaskMode("move_inside_mouth_stiffness")
                        run_once = False

                    forque_target_base = self.mouth_target_pose

                    closed_loop = False

                forque_source = self.getTransformationFromTF("base_link", "forque_end_effector")

                distance = np.linalg.norm(forque_source[:3,3] - forque_target_base[:3,3])

                intermediate_forque_target = np.zeros((4,4))
                intermediate_forque_target[0, 0] = 1
                intermediate_forque_target[1, 1] = 1
                intermediate_forque_target[2, 2] = 1
                intermediate_forque_target[:3,3] = np.array([0, 0, -distance]).reshape(1,3)
                intermediate_forque_target[3,3] = 1

                intermediate_forque_target = forque_target_base @ intermediate_forque_target

                intermediate_position_error = np.linalg.norm(forque_source[:3,3] - intermediate_forque_target[:3,3])
                intermediate_angular_error = self.getAngularDistance(forque_source[:3,:3], intermediate_forque_target[:3,:3])

                print("intermediate_position_error: ", forque_source[:3,3] - intermediate_forque_target[:3,3])
                print("intermediate_position_error mag: ", intermediate_position_error)
                print("intermediate_angular_error: ", intermediate_angular_error)

                ipe_forque_frame = np.linalg.inv(forque_source[:3,:3]) @ (forque_source[:3,3] - intermediate_forque_target[:3,3]).reshape(3,1)
                print("Error in forque frame: ",ipe_forque_frame)
                error_mag = np.linalg.norm(np.array([ipe_forque_frame[0], ipe_forque_frame[1]]))
                print("Error mag:",error_mag)
 
                if intermediate_position_error > INTERMEDIATE_THRESHOLD: # The thresholds here should be ideally larger than the thresholds for tracking trajectories
                    print("Tracking intermediate position... ")
                    target = self.getNextWaypoint(forque_source, intermediate_forque_target, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_XY)
                else:
                    print("Tracking target position... ")
                    distance_lookahead_update = INSIDE_DISTANCE_LOOKAHEAD_Z - intermediate_position_error
                    orientation_lookahead_update = ANGULAR_LOOKAHEAD - intermediate_angular_error
                    target = self.getNextWaypoint(intermediate_forque_target, forque_target_base, distance_lookahead = distance_lookahead_update)
                
                self.publishTaskCommand(target)
                self.publishTransformationToTF("base_link", "next_target", target)
                self.publishTransformationToTF("base_link", "final_target", forque_target_base)
                self.publishTransformationToTF("base_link", "intermediate_target", intermediate_forque_target)

            elif current_state == 3: # move outside mouth

                if closed_loop:

                    for i in range(0,10):

                        forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                        self.publishTaskCommand(forque_base)

                    time.sleep(0.1)

                    self.publishTaskMode("none")
                    self.publishTaskMode("move_outside_mouth_stiffness")

                    source_base = self.getTransformationFromTF("base_link", "forque_end_effector")

                    forque_target_source = np.identity(4)
                    forque_target_source[:3,3] = np.array([0, 0, -MOVE_OUTSIDE_DISTANCE]).reshape(1,3)

                    forque_target_base = source_base @ forque_target_source

                    closed_loop = False

                forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")

                target = self.getNextWaypoint(forque_base, forque_target_base, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_Z)
                
                self.publishTaskCommand(target)
                self.publishTransformationToTF("base_link", "next_target", target)
                self.publishTransformationToTF("base_link", "final_target", forque_target_base)
           
if __name__ == '__main__':

    bite_transfer_trajectory_tracker = BiteTransferTrajectoryTracker()
    bite_transfer_trajectory_tracker.runControlLoop()