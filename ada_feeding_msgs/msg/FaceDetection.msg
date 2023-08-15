# A message with the results of face detection on a single frame.

# Whether or not a face was detected in the image
bool is_face_detected
# The 3D position of the detected mouth center (stomion)
geometry_msgs/PointStamped detected_mouth_center
# Whether or not the mouth was detected to be open
bool is_mouth_open
