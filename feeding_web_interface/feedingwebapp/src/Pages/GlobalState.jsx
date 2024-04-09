/**
 * zustand is a state management system for React apps. It lets us manage global
 * state, i.e., state that persists beyond individual pages of the app. In
 * practice, it stores this state in cookies.
 */
import create from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * APP_PAGE specifies what page the app is on. It can take the following values:
 *  - Settings: the settings page, where the user can configure parameters of
 *    that dictate how the robot will behave.
 *  - Home: the main screen, where the robot will display its state, solicit
 *    user input, etc.
 */
export const APP_PAGE = {
  Settings: 'settings',
  Home: 'home'
}

/**
 * MEAL_STATE stores the current state of the meal, and configures the
 * functionality of APP_PAGE.Home. States pre-pended with "U_" are waiting
 * for user input whereas states pre-pended with "R_" are waiting for the
 * robot to finish moving. The states are:
 *   - U_PreMeal: Waiting for the user to click "Start Feeding."
 *   - R_MovingAbovePlate: Waiting for the robot to move above the plate.
 *   - U_BiteSelection: Waiting for the user to select the food item they want.
 *   - U_PlateLocator: Allows the user to teleoperate the robot to center the
 *     the plate.
 *   - R_BiteAcquisition: Waiting for the robot to execute one bite acquisition
 *     attempt.
 *   - R_MovingToRestingPosition: Waiting for the robot to move to resting
 *     position.
 *   - U_BiteAcquisitionCheck: Waiting for the user to specify whether the
 *     bite acquisition was succesful or not.
 *   - R_MovingToStagingConfiguration: Waiting for the robot to move to the
 *     staging configuration.
 *   - R_DetectingFace: Waiting for the robot to detect a face.
 *   - R_MovingToMouth: Waiting for the robot to finish moving to the user's
 *     mouth.
 *   - R_MovingFromMouth: Waiting for the robot to move
 *     from the user's mouth to the staging configuration. This is a separate
 *     action from R_MovingToStagingConfiguration since it is cartesian.
 *   - U_BiteDone: Waiting for the user to indicate that they are done eating
 *     the bite.
 *   - R_StowingArm: Waiting for the robot to stow the arm.
 *   - U_PostMeal: Waiting on the user to start another meal.
 */
export const MEAL_STATE = {
  U_PreMeal: 'U_PreMeal',
  R_MovingAbovePlate: 'R_MovingAbovePlate',
  U_BiteSelection: 'U_BiteSelection',
  U_PlateLocator: 'U_PlateLocator',
  R_BiteAcquisition: 'R_BiteAcquisition',
  R_MovingToRestingPosition: 'R_MovingToRestingPosition',
  U_BiteAcquisitionCheck: 'U_BiteAcquisitionCheck',
  R_MovingToStagingConfiguration: 'R_MovingToStagingConfiguration',
  R_DetectingFace: 'R_DetectingFace',
  R_MovingToMouth: 'R_MovingToMouth',
  R_MovingFromMouth: 'R_MovingFromMouth',
  U_BiteDone: 'U_BiteDone',
  R_StowingArm: 'R_StowingArm',
  U_PostMeal: 'U_PostMeal'
}

/**
 * SETTINGS_STATE controls which settings page to display.
 *  - MAIN: The main page, with options to navigate to the other pages.
 *  - BITE_TRANSFER: The bite transfer page, where the user can configure
 *    parameters for bite transfer.
 */
export const SETTINGS_STATE = {
  MAIN: 'MAIN',
  BITE_TRANSFER: 'BITE_TRANSFER'
}

// The name of the default parameter namespace
export const DEFAULT_NAMESPACE = 'default'

/**
 * useGlobalState is a hook to store and manipulate web app state that we want
 * to persist across re-renders and refreshes. It won't persist if cookies are
 * cleared.
 */
export const useGlobalState = create(
  persist(
    (set, get) => ({
      // The current app page
      appPage: APP_PAGE.Home,
      // The app's current meal state
      mealState: MEAL_STATE.U_PreMeal,
      // The app's previous meal state
      prevMealState: null,
      // The timestamp when the robot transitioned to its current meal state
      mealStateTransitionTime: Date.now(),
      // The currently displayed settings page
      settingsState: SETTINGS_STATE.MAIN,
      settingsPresets: { current: DEFAULT_NAMESPACE, customNames: [] },
      // The goal for the bite acquisition action, including the most recent
      // food item that the user selected in "bite selection"
      biteAcquisitionActionGoal: null,
      // The goal for the move to mouth action, including the most recent
      // message received from the face detection node where a
      // face was detected and within the distance bounds of the camera.
      moveToMouthActionGoal: null,
      // Last RobotMotion action response
      lastMotionActionResponse: null,
      // Whether or not the currently-executing robot motion was paused by the user
      paused: false,
      // Flag to indicate robot motion trough teleoperation interface
      teleopIsMoving: false,
      // Flag to indicate whether to auto-continue after face detection
      faceDetectionAutoContinue: true,
      // Flag to indicate whether to auto-continue in bite done after food-on-fork detection
      biteDoneAutoContinue: false,
      biteDoneAutoContinueSecs: 3.0,
      biteDoneAutoContinueProbThresh: 0.25,
      // Flags to indicate whether to auto-continue in bite acquisition check based on food-on-fork
      // detection
      biteAcquisitionCheckAutoContinue: false,
      biteAcquisitionCheckAutoContinueSecs: 3.0,
      biteAcquisitionCheckAutoContinueProbThreshLower: 0.25,
      biteAcquisitionCheckAutoContinueProbThreshUpper: 0.75,
      // Whether the settings bite transfer page is currently at the user's face
      // or not. This is in the off-chance that the mealState is not at the user's
      // face, the settings page is, and the user refreshes -- the page should
      // call MoveFromMouthToStaging instead of just MoveToStaging.
      biteTransferPageAtFace: false,
      // The button the user most recently clicked on the BiteDone page. In practice,
      // this is the state we transition to after R_MovingFromMouth. In practice,
      // it is either R_MovingAbovePlate, R_MovingToRestingPosition, or R_DetectingFace.
      mostRecentBiteDoneResponse: MEAL_STATE.R_DetectingFace,
      // How much the video on the Bite Selection page should be zoomed in.
      biteSelectionZoom: 1.0,

      // Setters for global state
      setAppPage: (appPage) =>
        set(() => ({
          appPage: appPage,
          settingsState: SETTINGS_STATE.MAIN,
          // Sometimes the settings menu leaves the robot in a paused state.
          // Thus, we reset it to an unpaused state.
          paused: false
        })),
      setMealState: (mealState, mostRecentBiteDoneResponse = null) =>
        set(() => {
          let retval = {
            mealState: mealState,
            prevMealState: get().mealState,
            mealStateTransitionTime: Date.now(),
            biteTransferPageAtFace: false // Reset this flag when the meal state changes
          }
          if (mostRecentBiteDoneResponse) {
            retval.mostRecentBiteDoneResponse = mostRecentBiteDoneResponse
          }
          return retval
        }),
      setSettingsState: (settingsState) =>
        set(() => ({
          settingsState: settingsState
        })),
      setSettingsPresets: (settingsPresets) =>
        set(() => ({
          settingsPresets: settingsPresets
        })),
      setBiteAcquisitionActionGoal: (biteAcquisitionActionGoal) =>
        set(() => ({
          biteAcquisitionActionGoal: biteAcquisitionActionGoal
        })),
      setLastMotionActionResponse: (lastMotionActionResponse) =>
        set(() => ({
          lastMotionActionResponse: lastMotionActionResponse
        })),
      setMoveToMouthActionGoal: (moveToMouthActionGoal) =>
        set(() => ({
          moveToMouthActionGoal: moveToMouthActionGoal
        })),
      setPaused: (paused) =>
        set(() => ({
          paused: paused
        })),
      setTeleopIsMoving: (teleopIsMoving) =>
        set(() => ({
          teleopIsMoving: teleopIsMoving
        })),
      setFaceDetectionAutoContinue: (faceDetectionAutoContinue) =>
        set(() => ({
          faceDetectionAutoContinue: faceDetectionAutoContinue
        })),
      setBiteDoneAutoContinue: (biteDoneAutoContinue) =>
        set(() => ({
          biteDoneAutoContinue: biteDoneAutoContinue
        })),
      setBiteDoneAutoContinueSecs: (biteDoneAutoContinueSecs) =>
        set(() => ({
          biteDoneAutoContinueSecs: biteDoneAutoContinueSecs
        })),
      setBiteDoneAutoContinueProbThresh: (biteDoneAutoContinueProbThresh) =>
        set(() => ({
          biteDoneAutoContinueProbThresh: biteDoneAutoContinueProbThresh
        })),
      setBiteAcquisitionCheckAutoContinue: (biteAcquisitionCheckAutoContinue) =>
        set(() => ({
          biteAcquisitionCheckAutoContinue: biteAcquisitionCheckAutoContinue
        })),
      setBiteAcquisitionCheckAutoContinueSecs: (biteAcquisitionCheckAutoContinueSecs) =>
        set(() => ({
          biteAcquisitionCheckAutoContinueSecs: biteAcquisitionCheckAutoContinueSecs
        })),
      setBiteAcquisitionCheckAutoContinueProbThreshLower: (biteAcquisitionCheckAutoContinueProbThreshLower) =>
        set(() => ({
          biteAcquisitionCheckAutoContinueProbThreshLower: biteAcquisitionCheckAutoContinueProbThreshLower
        })),
      setBiteAcquisitionCheckAutoContinueProbThreshUpper: (biteAcquisitionCheckAutoContinueProbThreshUpper) =>
        set(() => ({
          biteAcquisitionCheckAutoContinueProbThreshUpper: biteAcquisitionCheckAutoContinueProbThreshUpper
        })),
      setBiteTransferPageAtFace: (biteTransferPageAtFace) =>
        set(() => ({
          biteTransferPageAtFace: biteTransferPageAtFace
        })),
      setBiteSelectionZoom: (biteSelectionZoom) =>
        set(() => ({
          biteSelectionZoom: biteSelectionZoom
        }))
    }),
    { name: 'ada_web_app_global_state' }
  )
)
