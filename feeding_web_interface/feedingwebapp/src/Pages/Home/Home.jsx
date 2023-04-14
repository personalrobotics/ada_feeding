// React imports
import React from 'react'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'
import { useROS } from 'react-ros'

// Local imports
import './Home.css'
import { useGlobalState, MEAL_STATE } from '../GlobalState'
import BiteAcquisition from './MealStates/BiteAcquisition'
import BiteAcquisitionCheck from './MealStates/BiteAcquisitionCheck'
import BiteDone from './MealStates/BiteDone'
import BiteInitiation from './MealStates/BiteInitiation'
import BiteSelection from './MealStates/BiteSelection'
import MovingAbovePlate from './MealStates/MovingAbovePlate'
import MovingToMouth from './MealStates/MovingToMouth'
import MovingToStagingLocation from './MealStates/MovingToStagingLocation'
import PlateLocator from './MealStates/PlateLocator'
import PostMeal from './MealStates/PostMeal'
import PreMeal from './MealStates/PreMeal'
import StowingArm from './MealStates/StowingArm'

/**
 * Determines what screen to render based on the meal state.
 *
 * @param {MEAL_STATE} mealState - The current meal state. Must be one of the
 *        states specified in MEAL_STATE.
 * @param {bool} debug - Whether to run it in debug mode or not.
 */
function getComponentByMealState(mealState, debug) {
  console.log('getComponentByMealState', mealState, debug)
  switch (mealState) {
    case MEAL_STATE.U_PreMeal:
      return <PreMeal debug={debug} />
    case MEAL_STATE.R_MovingAbovePlate:
      return <MovingAbovePlate debug={debug} />
    case MEAL_STATE.U_BiteSelection:
      return <BiteSelection debug={debug} />
    case MEAL_STATE.U_PlateLocator:
      return <PlateLocator debug={debug} />
    case MEAL_STATE.R_BiteAcquisition:
      return <BiteAcquisition debug={debug} />
    case MEAL_STATE.U_BiteAcquisitionCheck:
      return <BiteAcquisitionCheck debug={debug} />
    case MEAL_STATE.R_MovingToStagingLocation:
      return <MovingToStagingLocation debug={debug} />
    case MEAL_STATE.U_BiteInitiation:
      return <BiteInitiation debug={debug} />
    case MEAL_STATE.R_MovingToMouth:
      return <MovingToMouth debug={debug} />
    case MEAL_STATE.U_BiteDone:
      return <BiteDone debug={debug} />
    case MEAL_STATE.R_StowingArm:
      return <StowingArm debug={debug} />
    case MEAL_STATE.U_PostMeal:
      return <PostMeal debug={debug} />
  }
}

/**
 * The Home component displays the state of the meal, solicits user input as
 * needed, and communicates with the robot (TODO (amaln)).
 */
function Home(props) {
  // Get the meal state
  const mealState = useGlobalState((state) => state.mealState)

  // useROS  gives us access to functions to configure and interact with ROS.
  // TODO (amaln): Actually connect this web app to ROS!
  let { /* ros,*/ isConnected /*, topics, services, toggleConnection, createListener */ } = useROS()

  // Render the component
  return (
    <div>
      {/**
       * Regardless of the mealState, always display whether the web app is
       * connected to the robot.
       */}
      {isConnected ? (
        <div>
          <p className='connectedDiv' style={{ fontSize: '24px' }}>
            🔌 connected
          </p>
        </div>
      ) : (
        <div>
          <p className='notConnectedDiv' style={{ fontSize: '24px' }}>
            ⛔ not connected
          </p>
        </div>
      )}

      {/**
       * The main contents of the screen depends on the mealState.
       */}
      {getComponentByMealState(mealState, props.debug)}
    </div>
  )
}
Home.propTypes = {
  debug: PropTypes.bool
}

export default Home
