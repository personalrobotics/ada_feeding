# This message is used to specify a single output of an image segmentation 
# algorithm.

# The axis-aligned bounding box that contains the segmented item
sensor_msgs/RegionOfInterest roi
# The binary mask, of the same size of the bounding box, that specifies which
# pixels in the bounding box are part of the segmented item
sensor_msgs/CompressedImage mask
# The RGB image that was used for segmentation, cropped to the roi.
sensor_msgs/CompressedImage rgb_image
# The depth image cropped to the roi.
sensor_msgs/Image depth_image

# The average of the depth camera readings over the mask (in m)
float64 average_depth

# An arbitrary ID that defines the segmented item
string item_id

# A score that indicates how confident the segemntation algorithm is in 
# this mask.
float64 confidence
