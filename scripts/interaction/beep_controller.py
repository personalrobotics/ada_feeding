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

TRIAL = 1

if __name__ == '__main__':

    rospy.init_node('beep_controller', anonymous=True)

    true_contact_type_publisher = rospy.Publisher('/true_contact_type', String, queue_size=10)
    track_preferred_bite_pose_publisher = rospy.Publisher('/track_preferred_bite_pose', Bool, queue_size=10)
    beep_publisher = rospy.Publisher('/beep', Float64, queue_size=10)

    if TRIAL == 1:
        stages = ["inside_intentional_tongue", "inside_intentional_bite"]
    else:
        stages = ["inside_incidental", "inside_intentional_bite"]

    time.sleep(0.1)
    true_contact_type_publisher.publish("outside_incidental")
    track_preferred_bite_pose_publisher.publish(False)

    for stage in stages:
        print("Stage: " + stage)
        print("Press enter to publish: ")
        lol = input()
        true_contact_type_publisher.publish(stage)
        beep_msg = Float64()
        beep_msg.data = 0.0
        beep_publisher.publish(beep_msg)
