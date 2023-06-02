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
 *   - R_MovingToMouth: Waiting for the robot to finish moving to the user's
 *     mouth.
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
  R_MovingToMouth: 'R_MovingToMouth',
  U_BiteDone: 'U_BiteDone',
  R_StowingArm: 'R_StowingArm',
  U_PostMeal: 'U_PostMeal'
}

/**
 * The parameters that users can set (keys) and a list of human-readable values
 * they can take on.
 *   - stagingPosition: Discrete options for where the robot should wait until
 *     the user is ready.
 *   - biteInitiation: Options for the modality the user wants to use to tell
 *     the robot they are ready for a bite.
 *       - TODO: Make these checkboxes instead -- users should be able to
 *         enable multiple buttons if they so desire.
 *   - biteSelection: Options for how the user wants to tell the robot what food
 *     item they want next.
 *
 * TODO (amaln): When we connect this to ROS, each of these settings types and
 * value options will have to have corresponding rosparam names and value options.
 */
export const SETTINGS = {
  stagingPosition: ['In Front of Me', 'On My Right Side'],
  biteInitiation: ['Open Mouth', 'Say "I am Ready"', 'Press Button'],
  biteSelection: ['Name of Food', 'Click on Food']
}

/**
 * useGlobalState is a hook to store and manipulate web app state that we want
 * to persist across re-renders and refreshes. It won't persist if cookies are
 * cleared.
 */
export const useGlobalState = create(
  persist(
    (set) => ({
      // The app's current meal state
      mealState: MEAL_STATE.U_PreMeal,
      // The timestamp when the robot transitioned to its current meal state
      mealStateTransitionTime: Date.now(),
      // The current app page
      appPage: APP_PAGE.Home,
      // The most recent food item that the user selected in "bite selection"
      desiredFoodItem: null,
      // Whether or not the currently-executing robot motion was paused by the user
      paused: false,
      // Settings values
      stagingPosition: SETTINGS.stagingPosition[0],
      biteInitiation: SETTINGS.biteInitiation[0],
      biteSelection: SETTINGS.biteSelection[0],

      // Setters for global state
      setMealState: (mealState) =>
        set(() => ({
          mealState: mealState,
          mealStateTransitionTime: Date.now()
        })),
      setAppPage: (appPage) =>
        set(() => ({
          appPage: appPage
        })),
      setDesiredFoodItem: (desiredFoodItem) =>
        set(() => ({
          desiredFoodItem: desiredFoodItem
        })),
      setPaused: (paused) =>
        set(() => ({
          paused: paused
        })),
      setStagingPosition: (stagingPosition) =>
        set(() => ({
          stagingPosition: stagingPosition
        })),
      setBiteInitiation: (biteInitiation) =>
        set(() => ({
          biteInitiation: biteInitiation
        })),
      setBiteSelection: (biteSelection) =>
        set(() => ({
          biteSelection: biteSelection
        }))
    }),
    { name: 'ada_web_app_global_state' }
  )
)
