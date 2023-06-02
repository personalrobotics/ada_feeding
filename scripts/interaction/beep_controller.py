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

class BeepController:
    def __init__(self):

        self.true_contact_type_publisher = rospy.Publisher('/true_contact_type', String, queue_size=10)
        self.contact_classification_pub =  rospy.Publisher("/contact_classification", String, queue_size=10)
        self.record_preferred_bite_pose_publisher = rospy.Publisher('/record_preferred_bite_pose', Bool, queue_size=10)
        self.beep_publisher = rospy.Publisher('/beep', Float64, queue_size=10)

        if TRIAL == 1:
            self.stages = ["inside_intentional_tongue", "inside_intentional_bite"]
        else:
            self.stages = ["inside_incidental", "inside_intentional_bite"]

        self.record_force_metrics = False
        self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ft_callback)
        self.ft_readings = []
        self.record_force_metrics_start_time = None

        print("Ready to publish")


    def ft_callback(self, msg):

        if self.record_force_metrics:
            ft_reading = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
            self.ft_readings.append(ft_reading)

    def run_control_loop(self):

        time.sleep(0.1)
        self.true_contact_type_publisher.publish("outside_incidental")

        print("Stage: " + self.stages[0])
        print("Press enter to beep and publish: ")
        
        lol = input()
        self.record_force_metrics = True
        self.true_contact_type_publisher.publish(self.stages[0])
        
        beep_msg = Float64()
        beep_msg.data = 0.0
        self.beep_publisher.publish(beep_msg)

        print("Stage: " + self.stages[1])
        print("Press enter to beep: ")
        lol = input()
        
        self.record_preferred_bite_pose_publisher.publish(True)
        self.record_force_metrics = False
        beep_msg = Float64()
        beep_msg.data = 0.0
        self.beep_publisher.publish(beep_msg)
        print("Press enter to publish: ")
        lol = input()
        self.true_contact_type_publisher.publish("outside_incidental")
        self.contact_classification_pub.publish(self.stages[1])

    def compile_metrics(self):

        self.ft_readings = np.array(self.ft_readings)
        print("Maximum Force Reading: ", np.max(np.linalg.norm(self.ft_readings, axis=1)))
        print("Mean Force Reading: ", np.mean(np.linalg.norm(self.ft_readings, axis=1)))
        print("Median Force Reading: ", np.median(np.linalg.norm(self.ft_readings, axis=1)))
        print("Variance Force Reading: ", np.var(np.linalg.norm(self.ft_readings, axis=1)))
        print("Energy Force Reading: ", np.sum(np.linalg.norm(self.ft_readings, axis=1)))

        print("\nprogram exiting gracefully")
        sys.exit(0)

if __name__ == '__main__':

    rospy.init_node('beep_controller', anonymous=True)
    beep_controller = BeepController()
    beep_controller.run_control_loop()
    beep_controller.compile_metrics()


    

   

    

    
