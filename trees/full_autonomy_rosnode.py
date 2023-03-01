from std_msgs.msg import String, Float64MultiArray, Bool
import tf 
import numpy as np
# service
from ada_feeding.srv import PlateService

# publishes through 3 topics
 def talker(locate_move):
    pub1 = rospy.Publisher('/alert', Bool, queue_size=10)
    pub2 = rospy.Publisher('/offset', Float64MultiArray, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    # convenient way for looping at the desired rate; here 10 times per second
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # alert topic publishes
        alert_and_offset_vals = alert_and_offset_client()
        rate = rospy.Rate(10.0)
        alert_val = alert_and_offset_vals[0]
        rospy.loginfo(alert_val)
        pub1.publish(alert_val)
        # frame transformations for offset
        listener = tf.TransformListener()
        try:
            (trans,rot) = listener.lookupTransform('camera_color_optical_frame', 'world', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        camera_offset = alert_and_offset_vals[1]
        robot_offset = np.cross(rot, camera_offset) + trans
        # offset topic publishes
        rospy.loginfo(robot_offset)
        pub2.publish(robot_offset)
        rate.sleep()

def callbackFn(data):
   rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
       
def alert_and_offset_client():
    rospy.wait_for_service('alert_and_offset')
    try:
        alert_and_offset = rospy.ServiceProxy('alert_and_offset', PlateService)
        vals = alert_and_offset()
        return vals
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def listener():
    rospy.init_node('listener', anonymous=True)
   
    rospy.Subscriber("/alert", Boolean, callbackFn)
    rospy.Subscriber("/offset", Float64MultiArray, callbackFn)
   
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
   
if __name__ == '__main__':
    try:
           talker()
       except rospy.ROSInterruptException:
           pass
    listener()