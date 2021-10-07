cmake_minimum_required(VERSION 3.10.0)
project(ada_feeding)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ===============================================================================
# Dependencies
#

find_package(tf_conversions REQUIRED)
include_directories(${tf_conversions_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})

find_package(DART 6.6.2 REQUIRED
  COMPONENTS utils utils-urdf optimizer-nlopt
)
include_directories(${DART_INCLUDE_DIRS})

find_package(aikido 0.3.0 REQUIRED
  COMPONENTS
    common
    constraint
    control
    control_ros
    distance
    perception
    planner
    planner_kunzretimer
    planner_ompl
    planner_parabolic
    planner_vectorfield
    planner_kunzretimer
    robot
    robot_ros
    statespace
    io
    rviz
)
include_directories(${aikido_INCLUDE_DIRS})

find_package(libada REQUIRED)
include_directories(${libada_INCLUDE_DIRS})

find_package(rewd_controllers)
IF (rewd_controllers_FOUND)
  include_directories(${rewd_controllers_INCLUDE_DIRS})
  message(STATUS "rewd controllers found!")
  add_definitions(-DREWD_CONTROLLERS_FOUND)
ELSE()
message("rewd controllers not found!")
ENDIF()

find_package(pr_assets REQUIRED)
include_directories(${pr_assets_INCLUDE_DIRS})

find_package(pr_tsr REQUIRED)
include_directories(${pr_tsr_INCLUDE_DIRS})

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosbag REQUIRED)
find_package(image_geometry REQUIRED)
find_package(sensor_msgs REQUIRED)

#==============================================================================
# Build Description
#

include_directories(include)

add_executable(feeding
  scripts/main.cpp
  scripts/demo.cpp
  scripts/humanStudy.cpp
  scripts/spanetDemo.cpp
  src/AcquisitionAction.cpp
  src/FoodItem.cpp
  src/FeedingDemo.cpp
  src/FTThresholdHelper.cpp
  src/Workspace.cpp
  src/util.cpp
  src/action/Grab.cpp
  src/action/DetectAndMoveAboveFood.cpp
  src/action/FeedFoodToPerson.cpp
  src/action/MoveAbove.cpp
  src/action/MoveAboveFood.cpp
  src/action/MoveAboveForque.cpp
  src/action/MoveAbovePlate.cpp
  src/action/MoveDirectlyToPerson.cpp
  src/action/MoveInFrontOfPerson.cpp
  src/action/MoveInto.cpp
  src/action/MoveOutOf.cpp
  src/action/MoveTowardsPerson.cpp
  src/action/PickUpFork.cpp
  src/action/PutDownFork.cpp
  src/action/Skewer.cpp
  src/perception/Perception.cpp
  src/perception/PerceptionServoClient.cpp
  src/ranker/ShortestDistanceRanker.cpp
  src/ranker/SuccessRateRanker.cpp
  src/ranker/TargetFoodRanker.cpp
)

include_directories(include $(rosbag_INCLUDE_DIRS))

target_link_libraries(feeding
  ${DART_LIBRARIES}
  ${aikido_LIBRARIES}
  ${Boost_LIBRARIES}
  ${tf_conversions_LIBRARIES}
  ${image_transport_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${rosbag_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${image_geometry_LIBRARIES}
  libada)

IF (rewd_controllers_FOUND)
  target_link_libraries(feeding ${rewd_controllers_LIBRARIES})
ENDIF()

install(TARGETS feeding RUNTIME DESTINATION bin)

# ===============================================================================
# Formatting
#
include(ClangFormat)
clang_format_setup(VERSION 6.0)

if(CLANG_FORMAT_EXECUTABLE)
  file(GLOB_RECURSE ALL_SOURCE_FILES
    LIST_DIRECTORIES false
    src/*.cpp
    include/feeding/*.hpp
  )

  clang_format_add_sources(${ALL_SOURCE_FILES})
  clang_format_add_targets()
endif()
