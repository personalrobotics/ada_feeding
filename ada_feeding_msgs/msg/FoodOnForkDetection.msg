# A message with the results of food on fork detection on a single frame.

# The header for the image the detection corresponds to
std_msgs/Header header

# The status of the food-on-fork detector.
int32 status
int32 SUCCESS=1
int32 ERROR_TOO_FEW_POINTS=-1
int32 UNKNOWN_ERROR=-99

# A probability in [0,1] that indicates the likelihood that there is food on the
# fork in the image. Only relevant if status == FoodOnForkDetection.SUCCESS
float64 probability

# Contains more details of the result, including any error messages that were encountered
string message
