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
 * A dictionary containing the icon associated with each "robot moving" state
 * except the Bite Acquisition state which has only a back button in footer.
 * This dictionary is used to populate buttons with the icon images for states
 * that those buttons will transition to.
 * The keys are the meal states where the robot moves.
 * Each key is paired with a value of an svg icon image of that meal state.
 */
let MOVING_STATE_ICON_DICT = {}
MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate] = '/robot_state_imgs/move_above_plate_position.svg'
MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToRestingPosition] = '/robot_state_imgs/move_to_resting_position.svg'
MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToStagingConfiguration] = '/robot_state_imgs/move_to_staging_configuration.svg'
MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingFromMouth] = '/robot_state_imgs/move_to_staging_configuration.svg'
MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth] = '/robot_state_imgs/move_to_mouth_position.svg'
MOVING_STATE_ICON_DICT[MEAL_STATE.R_StowingArm] = '/robot_state_imgs/stowing_arm_position.svg'
export { MOVING_STATE_ICON_DICT }

// The names of the ROS topic(s)
export const CAMERA_FEED_TOPIC = '/local/camera/color/image_raw/compressed'
export const FACE_DETECTION_TOPIC = '/face_detection'
export const FACE_DETECTION_TOPIC_MSG = 'ada_feeding_msgs/FaceDetection'
export const FACE_DETECTION_IMG_TOPIC = '/face_detection_img/compressed'
export const FOOD_ON_FORK_DETECTION_TOPIC = '/food_on_fork_detection'
export const FOOD_ON_FORK_DETECTION_TOPIC_MSG = 'ada_feeding_msgs/FoodOnForkDetection'
export const ROBOT_COMPRESSED_IMG_TOPICS = [CAMERA_FEED_TOPIC, FACE_DETECTION_IMG_TOPIC]
export const SERVO_CARTESIAN_TOPIC = '/web_app/servo_node/delta_twist_cmds'
export const SERVO_CARTESIAN_TOPIC_MSG = 'geometry_msgs/msg/TwistStamped'
export const SERVO_JOINT_TOPIC = '/web_app/servo_node/delta_joint_cmds'
export const SERVO_JOINT_TOPIC_MSG = 'control_msgs/msg/JointJog'

// States from which, if they fail, it is NOT okay for the user to retry the
// same action.
let NON_RETRYABLE_STATES = new Set()
NON_RETRYABLE_STATES.add(MEAL_STATE.R_BiteAcquisition)
export { NON_RETRYABLE_STATES }

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
ROS_ACTIONS_NAMES[MEAL_STATE.R_MovingToRestingPosition] = {
  actionName: 'MoveToRestingPosition',
  messageType: 'ada_feeding_msgs/action/MoveTo'
}
ROS_ACTIONS_NAMES[MEAL_STATE.R_MovingToStagingConfiguration] = {
  actionName: 'MoveToStagingConfiguration',
  messageType: 'ada_feeding_msgs/action/MoveTo'
}
ROS_ACTIONS_NAMES[MEAL_STATE.R_MovingFromMouth] = {
  actionName: 'MoveFromMouth',
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
export const ACTIVATE_CONTROLLER_ACTION_NAME = 'ActivateController'
export const ACTIVATE_CONTROLLER_ACTION_TYPE = 'ada_feeding_msgs/action/ActivateController'
export const CARTESIAN_CONTROLLER_NAME = 'jaco_arm_cartesian_controller'
export const JOINT_CONTROLLER_NAME = 'jaco_arm_servo_controller'
export const RECOMPUTE_WORKSPACE_WALLS_ACTION_NAME = 'recompute_workspace_walls'
export const RECOMPUTE_WORKSPACE_WALLS_ACTION_TYPE = 'ada_feeding_msgs/action/Trigger'

/**
 * For states that call ROS services, this dictionary contains
 * the service name and the message type
 */
let ROS_SERVICE_NAMES = {}
ROS_SERVICE_NAMES[MEAL_STATE.U_BiteSelection] = {
  serviceName: 'toggle_table_detection',
  messageType: 'std_srvs/srv/SetBool'
}
ROS_SERVICE_NAMES[MEAL_STATE.R_DetectingFace] = {
  serviceName: 'toggle_face_detection',
  messageType: 'std_srvs/srv/SetBool'
}
ROS_SERVICE_NAMES[MEAL_STATE.U_BiteDone] = {
  serviceName: 'toggle_food_on_fork_detection',
  messageType: 'std_srvs/srv/SetBool'
}
ROS_SERVICE_NAMES[MEAL_STATE.U_BiteAcquisitionCheck] = {
  serviceName: 'toggle_food_on_fork_detection',
  messageType: 'std_srvs/srv/SetBool'
}
export { ROS_SERVICE_NAMES }
export const CLEAR_OCTOMAP_SERVICE_NAME = 'clear_octomap'
export const CLEAR_OCTOMAP_SERVICE_TYPE = 'std_srvs/srv/Empty'
export const ACQUISITION_REPORT_SERVICE_NAME = 'ada_feeding_action_select/action_report'
export const ACQUISITION_REPORT_SERVICE_TYPE = 'ada_feeding_msgs/srv/AcquisitionReport'
export const GET_ROBOT_STATE_SERVICE_NAME = 'get_robot_state'
export const GET_ROBOT_STATE_SERVICE_TYPE = 'ada_feeding_msgs/srv/GetRobotState'
export const GET_PARAMETERS_SERVICE_NAME = 'ada_feeding_action_servers/get_parameters'
export const GET_PARAMETERS_SERVICE_TYPE = 'rcl_interfaces/srv/GetParameters'
export const SET_PARAMETERS_SERVICE_NAME = 'ada_feeding_action_servers/set_parameters_atomically'
export const SET_PARAMETERS_SERVICE_TYPE = 'rcl_interfaces/srv/SetParametersAtomically'

// The names of parameters users can change in the settings menu
export const DISTANCE_TO_MOUTH_PARAM = 'MoveToMouth.tree_kwargs.plan_distance_from_mouth'
export const ABOVE_PLATE_PARAM_JOINTS = 'MoveAbovePlate.tree_kwargs.joint_positions'
export const STAGING_PARAM_JOINTS = 'MoveToStagingConfiguration.tree_kwargs.goal_configuration'
export const STAGING_PARAM_POSITION = 'MoveFromMouth.tree_kwargs.staging_configuration_position'
export const STAGING_PARAM_ORIENTATION = 'MoveFromMouth.tree_kwargs.staging_configuration_quat_xyzw'
// TODO: Eventually, we should break AcquireFood into two actionss to avoid these
// two different resting parameters.
export const RESTING_PARAM_JOINTS_1 = 'AcquireFood.tree_kwargs.resting_joint_positions'
// TODO: We may need to remove the orientation constraint from the below action.
export const RESTING_PARAM_JOINTS_2 = 'MoveToRestingPosition.tree_kwargs.goal_configuration'

// Parameters for modifying the force threshold
export const FORCE_THRESHOLD_PARAM = 'wrench_threshold.fMag'
export const DEFAULT_FORCE_THRESHOLD = 1.0 // N
export const INCREASED_FORCE_THRESHOLD = 75.0 // N

// Robot link names
export const ROBOT_BASE_LINK = 'j2n6s200_link_base'
export const ROBOT_END_EFFECTOR = 'forkTip'
export const ROBOT_JOINTS = [
  'j2n6s200_joint_1',
  'j2n6s200_joint_2',
  'j2n6s200_joint_3',
  'j2n6s200_joint_4',
  'j2n6s200_joint_5',
  'j2n6s200_joint_6'
]

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

/**
 * A function that provides the text associated with robot motion for each meal state.
 */
export function getRobotMotionText(mealState) {
  switch (mealState) {
    case MEAL_STATE.R_MovingAbovePlate:
      return 'Waiting to move above the plate...'
    case MEAL_STATE.R_BiteAcquisition:
      return 'Waiting to acquire the food...'
    case MEAL_STATE.R_MovingToRestingPosition:
      return 'Waiting to move to the resting position...'
    case MEAL_STATE.R_MovingToStagingConfiguration:
      return 'Waiting to move in front of you...'
    case MEAL_STATE.R_MovingToMouth:
      return 'Waiting to move to your mouth...'
    case MEAL_STATE.R_MovingFromMouth:
      return 'Waiting to move from your mouth...'
    case MEAL_STATE.R_StowingArm:
      return 'Waiting to stow the arm...'
    default:
      return 'Unknown meal state' + mealState.toString()
  }
}

// Container IDs for multiple ToastContainers
export const REGULAR_CONTAINER_ID = 'RegularContainerID'
export const MODAL_CONTAINER_ID = 'ModalContainerID'
