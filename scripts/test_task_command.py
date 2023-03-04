#! /usr/bin/env python

import sys
import rospy
import actionlib
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CartesianTrajectoryPoint
import numpy as np
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


if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('jointgroup_test_py')

        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        broadcaster = tf2_ros.TransformBroadcaster()

        ee_poses = []
        ee_poses.append([0.3646871150489894, 0.09218768325965015, 0.6801714938930924, 0.11002658656523921, 0.658693737467407, 0.7170946519901976, 0.1994792484372193])
        # ee_poses.append([0.2383579038369039, -0.07867983091208824, 0.629233504580651, 0.06684675351340216, 0.6907275468130748, 0.7173349855341276, 0.06210866402574479])
        # ee_poses.append([0.24655212978762994, -0.049533089815867744, 0.6397319120872631, 0.058154261466631836, 0.6952115759903573, 0.7140210780723983, 0.058930862462753274])

        cmd_pub = rospy.Publisher('/task_space_compliant_controller/command', CartesianTrajectoryPoint, queue_size=10)
        

        for ee_pose in ee_poses:

            print("Press [ENTER] to send task command:")
            kk = input()

            cartesian_point = CartesianTrajectoryPoint()

            goal = Pose()
            goal.position.x = ee_pose[0]
            goal.position.y = ee_pose[1]
            goal.position.z = ee_pose[2]

            goal.orientation.x = ee_pose[3]
            goal.orientation.y = ee_pose[4]
            goal.orientation.z = ee_pose[5]
            goal.orientation.w = ee_pose[6]

            cartesian_point.point.pose = goal
            cmd_pub.publish(cartesian_point)

            while True:

                t = TransformStamped()

                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "base_link"
                t.child_frame_id = "final_target"

                t.transform.translation.x = ee_pose[0]
                t.transform.translation.y = ee_pose[1]
                t.transform.translation.z = ee_pose[2]

                t.transform.rotation.x = ee_pose[3]
                t.transform.rotation.y = ee_pose[4]
                t.transform.rotation.z = ee_pose[5]
                t.transform.rotation.w = ee_pose[6]

                broadcaster.sendTransform(t)

    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)