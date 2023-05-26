// React imports
import React, { useCallback, useEffect } from 'react'
import PropTypes from 'prop-types'

// Local imports
import './Home.css'
import { useGlobalState, MEAL_STATE } from '../GlobalState'
import BiteAcquisitionCheck from './MealStates/BiteAcquisitionCheck'
import BiteDone from './MealStates/BiteDone'
import BiteInitiation from './MealStates/BiteInitiation'
import BiteSelection from './MealStates/BiteSelection'
import PlateLocator from './MealStates/PlateLocator'
import PostMeal from './MealStates/PostMeal'
import PreMeal from './MealStates/PreMeal'
import RobotMotion from './MealStates/RobotMotion'
import { TIME_TO_RESET_MS } from '../Constants'

/**
 * The Home component displays the state of the meal, solicits user input as
 * needed, and communicates with the robot.
 *
 * @param {boolean} debug - whether to run it in debug mode (e.g., if you aren't
 *        simulatenously running the robot) or not
 * @param {string} webVideoServerURL - the URL of the web video server
 * @returns {JSX.Element}
 */
function Home(props) {
  // Get the relevant values from global state
  const mealState = useGlobalState((state) => state.mealState)
  const mealStateTransitionTime = useGlobalState((state) => state.mealStateTransitionTime)
  const setMealState = useGlobalState((state) => state.setMealState)
  const setPaused = useGlobalState((state) => state.setPaused)

  /**
   * Implement time-based transition of states. This is so that after the user
   * finishes a meal, when they start the next meal the app starts in PreMeal.
   * The `useEffect` with these parameters ensures that it is called when
   * reloading the page or when transitioning mealStates, but not when re-rendering.
   */
  useEffect(() => {
    if (Date.now() - mealStateTransitionTime >= TIME_TO_RESET_MS) {
      console.log('Reverting to PreMeal due to too much elapsed time in one state.')
      setMealState(MEAL_STATE.U_PreMeal)
      setPaused(false)
    }
  }, [mealStateTransitionTime, setMealState, setPaused])

  // Get the relevant global variables
  const desiredFoodItem = useGlobalState((state) => state.desiredFoodItem)
  const detectedMouthCenter = useGlobalState((state) => state.detectedMouthCenter)

  /**
   * Determines what screen to render based on the meal state.
   */
  const getComponentByMealState = useCallback(() => {
    console.log('getComponentByMealState', mealState, props.debug)
    switch (mealState) {
      case MEAL_STATE.U_PreMeal: {
        return <PreMeal debug={props.debug} />
      }
      case MEAL_STATE.R_MovingAbovePlate: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingAbovePlate
        let nextMealState = MEAL_STATE.U_BiteSelection
        let actionInput = {}
        let waitingText = 'Waiting for the robot to move above the plate...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={actionInput}
            waitingText={waitingText}
          />
        )
      }
      case MEAL_STATE.U_BiteSelection: {
        return <BiteSelection debug={props.debug} webVideoServerURL={props.webVideoServerURL} />
      }
      case MEAL_STATE.U_PlateLocator: {
        return <PlateLocator debug={props.debug} webVideoServerURL={props.webVideoServerURL} />
      }
      case MEAL_STATE.R_BiteAcquisition: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_BiteAcquisition
        let nextMealState = MEAL_STATE.U_BiteAcquisitionCheck
        let actionInput = { detected_food: desiredFoodItem }
        let waitingText = 'Waiting for the robot to acquire the food...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={actionInput}
            waitingText={waitingText}
          />
        )
      }
      case MEAL_STATE.U_BiteAcquisitionCheck: {
        return <BiteAcquisitionCheck debug={props.debug} />
      }
      case MEAL_STATE.R_MovingToStagingLocation: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingToStagingLocation
        let nextMealState = MEAL_STATE.U_BiteInitiation
        let actionInput = {}
        /**
         * TODO: Determine the right user-facing name to give this position.
         * "staging position"? "ready position?" what is most clear?
         */
        let waitingText = 'Waiting for the robot to move to the "ready" position...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={actionInput}
            waitingText={waitingText}
          />
        )
      }
      case MEAL_STATE.U_BiteInitiation: {
        return <BiteInitiation debug={props.debug} webVideoServerURL={props.webVideoServerURL} />
      }
      case MEAL_STATE.R_MovingToMouth: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingToMouth
        let nextMealState = MEAL_STATE.U_BiteDone
        let actionInput = { detected_mouth_center: detectedMouthCenter }
        let waitingText = 'Waiting for Robot to move to your mouth...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={actionInput}
            waitingText={waitingText}
          />
        )
      }
      case MEAL_STATE.U_BiteDone: {
        return <BiteDone debug={props.debug} />
      }
      case MEAL_STATE.R_StowingArm: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_StowingArm
        let nextMealState = MEAL_STATE.U_PostMeal
        let actionInput = {}
        let waitingText = 'Waiting for the robot to get out of your way...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={actionInput}
            waitingText={waitingText}
          />
        )
      }
      case MEAL_STATE.U_PostMeal: {
        return <PostMeal debug={props.debug} />
      }
      default: {
        return <div>Unknown meal state: {mealState}</div>
      }
    }
  }, [desiredFoodItem, detectedMouthCenter, mealState, props.debug, props.webVideoServerURL])

  // Render the component
  return (
    <div>
      {/**
       * The main contents of the screen depends on the mealState.
       */}
      {getComponentByMealState()}
    </div>
  )
}
Home.propTypes = {
  /**
   * Whether to run it in debug mode (e.g., if you aren't simulatenously running
   * the robot) or not
   */
  debug: PropTypes.bool,
  // The URL of the web video server
  webVideoServerURL: PropTypes.string.isRequired
}

export default Home
