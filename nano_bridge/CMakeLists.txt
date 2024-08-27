cmake_minimum_required(VERSION 3.8)
project(nano_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CompressedImage.msg"
  DEPENDENCIES sensor_msgs
)

# Install Python executables
install(PROGRAMS
  nano_bridge/sender.py
  nano_bridge/sender_compressed_image.py
  nano_bridge/receiver.py
  nano_bridge/receiver_compressed_image.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files.
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
