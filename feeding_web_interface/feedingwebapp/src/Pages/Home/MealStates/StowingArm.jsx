// React Imports
import React, { useState } from 'react'
import Button from 'react-bootstrap/Button'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'
import Row from 'react-bootstrap/Row'

// Local Imports
import Footer from '../../Footer/Footer'
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'

/**
 * The StowingArm component tells the user that the robot is currently
 * getting the arm out of their way. It waits for the robot to complete the
 * motion before moving to the next state.
 *
 * @params {object} props - contains any properties passed to this Component
 */
const StowingArm = (props) => {
  // Create a local state variable for whether the robot is paused.
  const [paused, setPaused] = useState(false)

  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)

  /**
   * Callback function for when the robot has finished stowing itself.
   */
  function stowingArmDone() {
    console.log('stowingArmDone')
    setMealState(MEAL_STATE.U_PostMeal)
  }

  /**
   * Callback function for when the back button is clicked.
   */
  const backMealState = MEAL_STATE.R_MovingAbovePlate
  function backCallback() {
    console.log('Back Clicked')
    setMealState(backMealState)
  }

  // Render the component
  return (
    <>
      {/* TODO: Consider vertically centering this element */}
      <Row className='justify-content-center mx-auto my-2 w-75'>
        <div>
          <h1 id={MEAL_STATE.R_MovingAbovePlate} className='waitingMsg'>
            Waiting for the robot to get out of your way...
          </h1>
          {props.debug ? (
            <Button variant='secondary' className='justify-content-center mx-2 mb-2' size='lg' onClick={stowingArmDone}>
              Continue (Debug Mode)
            </Button>
          ) : (
            <></>
          )}
        </div>
      </Row>
      {/**
       * Display the footer with the Pause button.
       */}
      <Footer
        pauseCallback={() => console.log('Pause Clicked')}
        backCallback={backCallback}
        backMealState={backMealState}
        resumeCallback={() => console.log('Resume Clicked')}
        paused={paused}
        setPaused={setPaused}
      />
    </>
  )
}
StowingArm.propTypes = {
  debug: PropTypes.bool.isRequired
}

export default StowingArm
