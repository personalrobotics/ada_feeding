# This message is used to specify an Acquisition Action based
# on the schema available at:
# https://drive.google.com/file/d/1IP74g9bTQAbf8lahd7nxqTfV1Aook3bG/view

### Reference Frames

## Food Frame:
# Origin: Center, Bottom of the food
# +Z = against gravity
# +X = major axis of food bounding ellipse

## Approach Frame
# Same as Food Frame, except:
# +X = -(pre_transform.translation())
# Projected onto Food X/Y plane

## Utensil Frame = "forkTip" TF

### Approach

# Utensil Initial Transform in Food Frame
geometry_msgs/Pose pre_transform

# Approach Offset from Food Origin
geometry_msgs/Vector3 pre_offset

# Approach Force/Torque Thresholds
float64 pre_force
float64 pre_torque

# Rotation Hint
# Food Frame has PI-rotation symmetry
# If set, choose the one closest to this orientation
# In World Frame
geometry_msgs/Vector3 pre_rot_hint

### Grasp

# In-Food Twist
# Linear (dX, dY, dZ) in *approach frame*
# Angular (omegaX, omegaY, omegaZ) in *utensil frame*
geometry_msgs/Twist grasp_twist
builtin_interfaces/Duration grasp_duration

# Grasp Force/Torque Thresholds
float64 grasp_force
float64 grasp_torque

### Extraction

# Extraction Twist
# Linear (dX, dY, dZ) in *approach frame*
# Angular (omegaX, omegaY, omegaZ) in *utensil frame*
geometry_msgs/Twist ext_twist
builtin_interfaces/Duration ext_duration

# Extraction Force/Torque Thresholds
float64 ext_force
float64 ext_torque
