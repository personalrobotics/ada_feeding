// React imports
import React, { useCallback, useEffect, useMemo } from 'react'
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local imports
import './Home.css'
import { useGlobalState, MEAL_STATE } from '../GlobalState'
import BiteAcquisitionCheck from './MealStates/BiteAcquisitionCheck'
import BiteDone from './MealStates/BiteDone'
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

  /**
   * All action inputs are constant. Note that we must be cautious if making
   * them non-constant, because the robot will re-execute an action every time
   * the action input changes (even on re-renders).
   */
  const moveAbovePlateActionInput = useMemo(() => ({}), [])
  const biteAcquisitionActionInput = useMemo(() => desiredFoodItem, [desiredFoodItem])
  const moveToRestingPositionActionInput = useMemo(() => ({}), [])
  const moveToMouthActionInput = useMemo(() => ({}), [])
  const moveToStowPositionActionInput = useMemo(() => ({}), [])

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
        let waitingText = 'Waiting to move above the plate...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={moveAbovePlateActionInput}
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
        let waitingText = 'Waiting to acquire the food...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={biteAcquisitionActionInput}
            waitingText={waitingText}
          />
        )
      }
      case MEAL_STATE.R_MovingToRestingPosition: {
        let currentMealState = MEAL_STATE.R_MovingToRestingPosition
        let nextMealState = MEAL_STATE.U_BiteAcquisitionCheck
        let waitingText = 'Waiting to move to the resting position...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={moveToRestingPositionActionInput}
            waitingText={waitingText}
          />
        )
      }
      case MEAL_STATE.U_BiteAcquisitionCheck: {
        return <BiteAcquisitionCheck debug={props.debug} />
      }
      case MEAL_STATE.R_MovingToMouth: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingToMouth
        let nextMealState = MEAL_STATE.U_BiteDone
        let waitingText = 'Waiting to move to your mouth...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={moveToMouthActionInput}
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
        let waitingText = 'Waiting to get out of your way...'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            nextMealState={nextMealState}
            actionInput={moveToStowPositionActionInput}
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
  }, [
    mealState,
    props.debug,
    props.webVideoServerURL,
    biteAcquisitionActionInput,
    moveAbovePlateActionInput,
    moveToMouthActionInput,
    moveToRestingPositionActionInput,
    moveToStowPositionActionInput
  ])

  // Render the component
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      {/**
       * The main contents of the screen depends on the mealState.
       */}
      {getComponentByMealState()}
    </View>
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
