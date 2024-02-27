# A message with the results of food on fork detection on a single frame.

# The header for the image the detection corresponds to
std_msgs/Header header

# A probability in [0,1] that indicates the likelihood that there is food on the
# fork in the image. Negative values indicate error.
float64 probability

# If an error was encountered, this field will contain the error message.
string message
