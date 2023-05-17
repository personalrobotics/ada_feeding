import { MEAL_STATE } from './GlobalState'

// How often to check if ROS is connected
export const ROS_CHECK_INTERVAL_MS = 1000

// The RealSense's default video stream is 640x480
export const REALSENSE_WIDTH = 640
export const REALSENSE_HEIGHT = 480

/**
 * If the app has not transitioned states in this amount of time, it will reset
 * to PreMeal on the next rendering.
 */
export const TIME_TO_RESET_MS = 3600000 // 1 hour in milliseconds

/**
 * A dictionary containing the icon associated with each "robot motion" state.
 * The keys are the meal states where the robot moves.
 * Each key is paired with a value of an svg icon image of that meal state.
 */
let FOOTER_STATE_ICON_DICT = {}
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate] = '/robot_state_imgs/move_above_plate_position.svg'
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_BiteAcquisition] = '/robot_state_imgs/move_to_bite_acquisition_position.svg'
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingToStagingLocation] = '/robot_state_imgs/move_to_staging_position.svg'
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth] = '/robot_state_imgs/move_to_mouth_position.svg'
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_StowingArm] = '/robot_state_imgs/stowing_arm_position.svg'
export { FOOTER_STATE_ICON_DICT }

// The names of the ROS topic(s)
export const CAMERA_FEED_TOPIC = '/camera/color/image_raw'
export const FACE_DETECTION_TOPIC = '/face_detection'
export const FACE_DETECTION_TOPIC_MSG = 'ada_feeding_msgs/FaceDetection'
export const FACE_DETECTION_IMG_TOPIC = '/face_detection_img'

/**
 * For states that call ROS actions, this dictionary contains
 * the action name and the message type
 */
let ROS_ACTIONS_NAMES = {}
ROS_ACTIONS_NAMES[MEAL_STATE.R_MovingAbovePlate] = {
  actionName: 'MoveAbovePlate',
  messageType: 'ada_feeding_msgs/action/MoveTo'
}
ROS_ACTIONS_NAMES[MEAL_STATE.U_BiteSelection] = {
  actionName: 'SegmentFromPoint',
  messageType: 'ada_feeding_msgs/action/SegmentFromPoint'
}
ROS_ACTIONS_NAMES[MEAL_STATE.R_BiteAcquisition] = {
  actionName: 'AcquireFood',
  messageType: 'ada_feeding_msgs/action/AcquireFood'
}
ROS_ACTIONS_NAMES[MEAL_STATE.R_MovingToStagingLocation] = {
  actionName: 'MoveToStagingLocation',
  messageType: 'ada_feeding_msgs/action/MoveTo'
}
ROS_ACTIONS_NAMES[MEAL_STATE.R_MovingToMouth] = {
  actionName: 'MoveToMouth',
  messageType: 'ada_feeding_msgs/action/MoveToMouth'
}
ROS_ACTIONS_NAMES[MEAL_STATE.R_StowingArm] = {
  actionName: 'MoveToStowLocation',
  messageType: 'ada_feeding_msgs/action/MoveTo'
}
export { ROS_ACTIONS_NAMES }

/**
 * For states that call ROS services, this dictionary contains
 * the service name and the message type
 */
let ROS_SERVICE_NAMES = {}
ROS_SERVICE_NAMES[MEAL_STATE.U_BiteInitiation] = {
  serviceName: 'ToggleFaceDetection',
  messageType: 'ada_feeding_msgs/srv/ToggleFaceDetection'
}
export { ROS_SERVICE_NAMES }

/**
 * The meaning of the status that motion actions return in their results.
 * These should match the action definition(s).
 */
export const MOTION_STATUS_SUCCESS = 0
export const MOTION_STATUS_PLANNING_FAILED = 1
export const MOTION_STATUS_MOTION_FAILED = 2
export const MOTION_STATUS_CANCELED = 3
export const MOTION_STATUS_UNKNOWN = 99

/**
 * The meaning of the status that segmentation actions return in their results.
 * These should match the action definition(s).
 */
export const SEGMENTATION_STATUS_SUCCESS = 0
export const SEGMENTATION_STATUS_FAILED = 1
export const SEGMENTATION_STATUS_CANCELED = 3
export const SEGMENTATION_STATUS_UNKNOWN = 99

/**
 * The meaning of ROS Action statuses.
 * https://docs.ros2.org/latest/api/rclpy/api/actions.html#rclpy.action.server.GoalEvent
 */
export const ROS_ACTION_STATUS_EXECUTE = '1'
export const ROS_ACTION_STATUS_CANCEL_GOAL = '2'
export const ROS_ACTION_STATUS_SUCCEED = '3'
export const ROS_ACTION_STATUS_ABORT = '4'
export const ROS_ACTION_STATUS_CANCELED = '5'
