import { MEAL_STATE } from './GlobalState'
// The RealSense's default video stream is 640x480
export const REALSENSE_WIDTH = 640
export const REALSENSE_HEIGHT = 480

/**
 * If the app has not transitioned states in this amount of time, it will reset
 * to PreMeal on the next rendering.
 */
export const TIME_TO_RESET_MS = 3600000 // 1 hour in milliseconds

/**
 * A dictionary containing icon images for the back and resume buttons of the footer.
 * The keys are the meal states where the robot moves.
 * Each key is paired with a value of an svg icon image of that meal state.
 * TODO: This dictionary works well for the "Resume" button, but not for the "Back"
 * button which sometimes may need to go back further than the immediately-previous state.
 * This needs to be re-visited when implementing the "Back" button.
 */
let FOOTER_STATE_ICON_DICT = {}
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate] = '/robot_state_imgs/move_above_plate_position.svg'
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_BiteAcquisition] = '/robot_state_imgs/move_to_bite_acquisition_position.svg'
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingToStagingLocation] = '/robot_state_imgs/move_to_staging_position.svg'
FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth] = '/robot_state_imgs/move_to_mouth_position.svg'
export { FOOTER_STATE_ICON_DICT }
