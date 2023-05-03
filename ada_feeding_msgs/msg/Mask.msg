# This message is used to specify a single output of an image segmentation 
# algorithm.

# The axis-aligned bounding box that contains the segmented item
sensor_msgs/RegionOfInterest roi
# The binary mask, of the same size of the bounding box, that specifies which
# pixels in the bounding box are part of the segmented item
sensor_msgs/Image mask
# An arbitrary ID that defines the segmented item
string food_id
# A score that indicates how confident the segemntation algorithm is in 
# this mask.
float64 confidence