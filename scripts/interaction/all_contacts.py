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
from std_msgs.msg import Int64, Float64MultiArray
from std_msgs.msg import String, Float64, Bool
import threading
import time
import signal

# Parameters
OPEN_LOOP_RADIUS = 0.02
# OPEN_LOOP_RADIUS = 0.0
INTERMEDIATE_THRESHOLD_RELAXED = 0.02
INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED = 5*np.pi/180
INTERMEDIATE_THRESHOLD = 0.014
INFRONT_DISTANCE_LOOKAHEAD = 0.04
INSIDE_DISTANCE_LOOKAHEAD_Z = 0.045
INSIDE_DISTANCE_LOOKAHEAD_XY = 0.025
TILT_DISTANCE_LOOKAHEAD = 0.05
TILT_ANGULAR_LOOKAHEAD = 10*np.pi/180
ANGULAR_LOOKAHEAD = 5*np.pi/180
MOVE_OUTSIDE_DISTANCE = 0.1
TILT_MOVE_OUTSIDE_DISTANCE = 0.14
MIN_DURATION_MOUTH_OPEN = 1.0
IMPULSE_RETRACTION_DISTANCE = 0.1
IMPULSE_VELOCITY_THRESHOLD = 0.15
DISTANCE_INFRONT_MOUTH = 0.12

TRIAL = 1

class BiteTransferTrajectoryTracker:
    def __init__(self):

        rospy.init_node('jointgroup_test_py')
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.tongue_detected = False
        self.bite_detected = False
        self.impulse_detected = False
        self.contact_classification_sub = rospy.Subscriber('/contact_classification', String, self.contactClassificationCallback)

        self.update_preferred_bite_pose = False
        self.record_preferred_bite_pose_publisher = rospy.Publisher('/record_preferred_bite_pose', Bool, queue_size=10)
        self.track_preferred_bite_pose_publisher = rospy.Publisher('/track_preferred_bite_pose', Bool, queue_size=10)

        self.impulse_start_time = None
        self.switched_goal_time = None

        self.task_cmd_publisher = rospy.Publisher('/task_space_compliant_controller/command', CartesianTrajectoryPoint, queue_size=10)
        self.task_mode_publisher = rospy.Publisher('/task_space_compliant_controller/mode', String, queue_size=10)

        self.control_rate = rospy.Rate(100.0)

        self.state = 0
        self.state_lock = threading.Lock()

        self.last_mouth_closed_time = time.time()
        self.mouth_state_sub = rospy.Subscriber('/head_perception/mouth_state', Bool, self.mouth_state_callback) 

        self.state_sub = rospy.Subscriber('/state', Int64, self.state_callback)

        self.initial_head_pose = None
        self.final_head_pose = None

        self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ft_callback)

        self.maximum_ft_reading = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.paused_once = False

    def ft_callback(self, msg):

        ft_reading = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        mag = np.linalg.norm(ft_reading)
        if mag > np.linalg.norm(self.maximum_ft_reading):
            self.maximum_ft_reading = ft_reading
    
    def state_callback(self, msg):

        with self.state_lock:
            self.state = msg.data

    def mouth_state_callback(self, msg):

        current_state = None
        with self.state_lock:
            current_state = self.state

        if current_state == 0 and msg.data:
            with self.state_lock:
                self.state = 1

    def contactClassificationCallback(self, msg):

        if not self.tongue_detected and msg.data == "inside_intentional_tongue":
            self.tongue_detected = True
            with self.state_lock:
                self.state = 5

        if not self.bite_detected and msg.data == "inside_intentional_bite":
            self.bite_detected = True
            with self.state_lock:
                self.state = 3

        if not self.impulse_detected and msg.data == "inside_incidental":
            self.impulse_detected = True
            with self.state_lock:
                self.state = 6

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
                # print("Looking for transform")
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
        paused_once = False
        switched_goal = False

        # Assumption: No one will be updating state when this runs
        previous_state = self.state
        current_state = self.state

        last_time = time.time()
        while True: 

            self.control_rate.sleep()

            # print("Frequency: ",1.0/(time.time() - last_time))
            last_time = time.time()
            
            with self.state_lock:
                current_state = self.state

            if current_state != previous_state:
                print("Switching to self.state: ",current_state)
                closed_loop = True
                run_once = True
                switched_goal = False
                previous_state = current_state

            # print("Current self.state: ",current_state)

            # forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
            # print("Forque base: ",forque_base)

            if current_state == -2: # lurch back

                if run_once:
                    
                    forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                    shift_back = np.identity(4)
                    shift_back[:3,3] = np.array([0, 0, -IMPULSE_RETRACTION_DISTANCE]).reshape(1,3)

                    forque_target = forque_base @ shift_back

                    for i in range(0,10):
                        self.publishTaskCommand(forque_target)

                    run_once = False

                forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                distance = np.linalg.norm(forque_base[:3,3] - forque_target[:3,3])

                if distance < 0.05:
                    with self.state_lock:
                        self.state = -1


            if current_state == -1: # maintain position

                if run_once:
                    for i in range(0,10):

                        forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                        self.publishTaskCommand(forque_base)
                    run_once = False

            if current_state == 1: # move to infront of mouth
                
                # trajectory positions
                if closed_loop:

                    if run_once:

                        for i in range(0,10):

                            forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                            self.publishTaskCommand(forque_base)

                        time.sleep(0.1)

                        self.publishTaskMode("none")
                        self.publishTaskMode("default_stiffness")
                        run_once = False

                    forque_target_base = self.getTransformationFromTF("base_link", "forque_end_effector_target")

                    # forque_target_base = np.array([[-0.90093711, -0.1803962, 0.39467648, 0.42479337], 
                    #     [ 0.41205567, -0.07038973,  0.90843569,  0.16312421],
                    #     [-0.13609718, 0.98107212, 0.13775, 0.69508812],
                    #     [ 0., 0., 0., 1.]])

                    closed_loop = False

                    servo_point_forque_target = np.identity(4)
                    servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)

                    servo_point_base = forque_target_base @ servo_point_forque_target

                # current position
                forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")

                distance = np.linalg.norm(forque_base[:3,3] - servo_point_base[:3,3])
                angular_distance = self.getAngularDistance(forque_base[:3,:3], servo_point_base[:3,:3])

                # print("Distance: {} Angular Distance: {}".format(distance, angular_distance))
                # print("Threshold: {} Angular Threshold: {}".format(INTERMEDIATE_THRESHOLD_RELAXED, INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED))

                if distance < INTERMEDIATE_THRESHOLD_RELAXED and angular_distance < INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED:
                    with self.state_lock:
                        self.state = 2

                target = self.getNextWaypoint(forque_base, servo_point_base, distance_lookahead=INFRONT_DISTANCE_LOOKAHEAD)
                # target = servo_point_base

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

                        print("Using pose integral and inside mouth stiffness")

                        self.publishTaskMode("use_pose_integral")
                        self.publishTaskMode("move_inside_mouth_stiffness")
                        run_once = False

                        if not paused_once:
                            paused_once = True
                            time.sleep(0.5)
                            print("PAUSING")
                        
                    forque_target_base = self.getTransformationFromTF("base_link", "forque_end_effector_target")

                    # forque_target_base = np.array([[-0.90093711, -0.1803962, 0.39467648, 0.42479337], 
                    #     [ 0.41205567, -0.07038973,  0.90843569,  0.16312421],
                    #     [-0.13609718, 0.98107212, 0.13775, 0.69508812],
                    #     [ 0., 0., 0., 1.]])

                    # closed_loop = False

                forque_source = self.getTransformationFromTF("base_link", "forque_end_effector")

                distance = np.linalg.norm(forque_source[:3,3] - forque_target_base[:3,3])

                if distance < OPEN_LOOP_RADIUS and closed_loop:
                    if TRIAL == 2:
                        # if switched_goal:
                        #     if time.time() - self.switched_goal_time > 0.5:
                        #         print("Switching to open loop")
                        #         closed_loop = False
                        # else:
                        print("Switching to tracking preferred bite pose")
                        # switched_goal = True
                        # self.switched_goal_time = time.time()
                        msg = Bool()
                        msg.data = True
                        self.track_preferred_bite_pose_publisher.publish(msg)
                        time.sleep(0.5)
                        with self.state_lock:
                            self.state = 7
                    else:
                        closed_loop = False
                    
                    print(" Recording initial head pose...")
                    self.initial_head_pose = self.getTransformationFromTF("base_link", "head_pose")

                intermediate_forque_target = np.zeros((4,4))
                intermediate_forque_target[0, 0] = 1
                intermediate_forque_target[1, 1] = 1
                intermediate_forque_target[2, 2] = 1
                intermediate_forque_target[:3,3] = np.array([0, 0, -distance]).reshape(1,3)
                intermediate_forque_target[3,3] = 1

                intermediate_forque_target = forque_target_base @ intermediate_forque_target

                intermediate_position_error = np.linalg.norm(forque_source[:3,3] - intermediate_forque_target[:3,3])
                intermediate_angular_error = self.getAngularDistance(forque_source[:3,:3], intermediate_forque_target[:3,:3])

                # print("closed_loop: ",closed_loop)
                # print("intermediate_position_error: ", forque_source[:3,3] - intermediate_forque_target[:3,3])
                # print("intermediate_position_error mag: ", intermediate_position_error)
                # print("INTERMEDIATE_THRESHOLD mag: ", INTERMEDIATE_THRESHOLD)
                # print("intermediate_angular_error: ", intermediate_angular_error)

                ipe_forque_frame = np.linalg.inv(forque_source[:3,:3]) @ (forque_source[:3,3] - intermediate_forque_target[:3,3]).reshape(3,1)
                # print("Error in forque frame: ",ipe_forque_frame)
                error_mag = np.linalg.norm(np.array([ipe_forque_frame[0], ipe_forque_frame[1]]))
                # print("Error mag:",error_mag)
 
                if intermediate_position_error > INTERMEDIATE_THRESHOLD: # The thresholds here should be ideally larger than the thresholds for tracking trajectories
                    # print("Tracking intermediate position... ")
                    target = self.getNextWaypoint(forque_source, intermediate_forque_target, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_XY)
                else:
                    # print("Tracking target position... ")
                    distance_lookahead_update = INSIDE_DISTANCE_LOOKAHEAD_Z - intermediate_position_error
                    orientation_lookahead_update = ANGULAR_LOOKAHEAD - intermediate_angular_error
                    target = self.getNextWaypoint(intermediate_forque_target, forque_target_base, distance_lookahead = distance_lookahead_update)
                
                self.publishTaskCommand(target)
                self.publishTransformationToTF("base_link", "next_target", target)
                self.publishTransformationToTF("base_link", "final_target", forque_target_base)
                self.publishTransformationToTF("base_link", "intermediate_target", intermediate_forque_target)

            elif current_state == 3: # move outside mouth

                if closed_loop:
                    
                    print(" Recording final head pose...")
                    self.final_head_pose = self.getTransformationFromTF("base_link", "head_pose")

                    for i in range(0,10):

                        forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                        self.publishTaskCommand(forque_base)

                    time.sleep(0.1)

                    self.publishTaskMode("none")
                    self.publishTaskMode("move_outside_mouth_stiffness")
                    

                    if self.update_preferred_bite_pose:
                        msg = Bool()
                        msg.data = True
                        self.record_preferred_bite_pose_publisher.publish(msg)

                    if TRIAL == 2:
                        msg = Bool()
                        msg.data = False
                        self.track_preferred_bite_pose_publisher.publish(msg)
                    
                    time.sleep(0.5)

                    source_base = self.getTransformationFromTF("base_link", "forque_end_effector")

                    forque_target_source = np.identity(4)
                    forque_target_source[:3,3] = np.array([0, 0, -MOVE_OUTSIDE_DISTANCE]).reshape(1,3)

                    forque_target_base = source_base @ forque_target_source

                    closed_loop = False

                forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")

                distance = np.linalg.norm(forque_base[:3,3] - forque_target_base[:3,3])

                if distance < INTERMEDIATE_THRESHOLD_RELAXED:
                    with self.state_lock:
                        self.state = 4

                target = self.getNextWaypoint(forque_base, forque_target_base, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_Z)
                
                self.publishTaskCommand(target)
                self.publishTransformationToTF("base_link", "next_target", target)
                self.publishTransformationToTF("base_link", "final_target", forque_target_base)

            elif current_state == 4: # move to fixed position and exit

                if run_once:

                    for i in range(0,10):

                            forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                            self.publishTaskCommand(forque_base)

                    time.sleep(0.1)

                    self.publishTaskMode("none")
                    self.publishTaskMode("default_stiffness")
                    run_once = False
                
                final_target = np.array([[-0.72820993, -0.01297005,  0.6852314,   0.2797607 ], 
                                            [ 0.68525268, -0.03098182,  0.72764613, -0.13926232], 
                                            [ 0.01179211,  0.99943579,  0.03144903,  0.55620243],
                                            [ 0.,          0.,          0.,          1.        ]])
                
                # current position
                forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                
                target = self.getNextWaypoint(forque_base, final_target, distance_lookahead=INFRONT_DISTANCE_LOOKAHEAD)

                self.publishTaskCommand(target)
                self.publishTransformationToTF("base_link", "next_target", target)
                self.publishTransformationToTF("base_link", "final_target", final_target)

            elif current_state == 5: # switch to C3

                if run_once:

                    for i in range(0,10):

                            forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                            self.publishTaskCommand(forque_base)

                    time.sleep(0.1)

                    print("Publishing zero")
                    self.publishTaskMode("zero_contact")
                    
                    self.update_preferred_bite_pose = True

                    run_once = False

            elif current_state == 6: # switch to C3 - impulse

                if run_once:

                    msg = Bool()
                    msg.data = False
                    self.track_preferred_bite_pose_publisher.publish(msg)

                    for i in range(0,10):

                            forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                            self.publishTaskCommand(forque_base)

                    # time.sleep(0.1)

                    print("Publishing zero")
                    self.publishTaskMode("zero_contact")
                    
                    self.update_preferred_bite_pose = False

                    run_once = False
                    self.impulse_start_time = time.time()

                if time.time() - self.impulse_start_time > 5.0:
                    print("Impulse over")
                    with self.state_lock:
                        self.state = 2

            elif current_state == 7: # track position

                if closed_loop:

                    if run_once:

                        for i in range(0,10):

                            forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                            self.publishTaskCommand(forque_base)

                    time.sleep(0.1)

                    self.publishTaskMode("none")
                    self.publishTaskMode("high_stiffness")
                    run_once = False
                        
                    forque_target_base = self.getTransformationFromTF("base_link", "forque_end_effector_target")

                forque_source = self.getTransformationFromTF("base_link", "forque_end_effector")

                distance = np.linalg.norm(forque_source[:3,3] - forque_target_base[:3,3])

                if distance < OPEN_LOOP_RADIUS and closed_loop:
                    closed_loop = False

                target = self.getNextWaypoint(forque_base, forque_target_base, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_XY)
                
                self.publishTaskCommand(target)
                self.publishTransformationToTF("base_link", "next_target", target)
                self.publishTransformationToTF("base_link", "final_target", forque_target_base)              

                



    def signal_handler(self, signal, frame):

        print("Maximum Force Reading: ", self.maximum_ft_reading)

        if self.initial_head_pose is not None and self.final_head_pose is not None:
            print("Initial Head Pose: ", self.initial_head_pose)
            print("Final Head Pose: ", self.final_head_pose)

            self.publishTransformationToTF("base_link", "initial_head_pose", self.initial_head_pose)
            self.publishTransformationToTF("base_link", "final_head_pose", self.final_head_pose)

            neck_flexion, neck_rotation, neck_lateral_flexion = Rotation.from_matrix(self.final_head_pose[:3,:3]).as_euler('xyz')
            reference_neck_flexion, reference_neck_rotation, reference_neck_lateral_flexion = Rotation.from_matrix(self.initial_head_pose[:3,:3]).as_euler('xyz')

            translation_from_reference = (self.final_head_pose[:3,3] - self.initial_head_pose[:3,3]).reshape(3,)
            rotation_from_reference = np.array([neck_flexion - reference_neck_flexion, neck_rotation - reference_neck_rotation, neck_lateral_flexion - reference_neck_lateral_flexion])
            
            print("Head Movement: ")
            print("[x, y, z]: ",translation_from_reference)
            print("[flexion, rotation, lateral_flexion]: ",rotation_from_reference)

        print("\nprogram exiting gracefully")
        sys.exit(0)

if __name__ == '__main__':

    bite_transfer_trajectory_tracker = BiteTransferTrajectoryTracker()
    signal.signal(signal.SIGINT, bite_transfer_trajectory_tracker.signal_handler) # ctrl+c
    
    bite_transfer_trajectory_tracker.runControlLoop()

    rospy.spin()