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
 * The BiteAcquisition component tells the user that the robot is currently
 * acquiring the bite. It waits for the robot to complete the motion before
 * moving to the next state.
 *
 * @params {object} props - contains any properties passed to this Component
 */
const BiteAcquisition = (props) => {
  // Create a local state variable for whether the robot is paused.
  const [paused, setPaused] = useState(false)

  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  const desiredFoodItem = useGlobalState((state) => state.desiredFoodItem)
  console.log(desiredFoodItem)

  /**
   * Callback function for when the bite acquisition is done.
   */
  function biteAcquisitionDone() {
    console.log('biteAcquisitionDone')
    setMealState(MEAL_STATE.U_BiteAcquisitionCheck)
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
            Waiting for the robot to acquire the food...
          </h1>
          {props.debug ? (
            <Button variant='secondary' className='justify-content-center mx-2 mb-2' size='lg' onClick={biteAcquisitionDone}>
              Continue (Debug Mode)
            </Button>
          ) : (
            <></>
          )}
        </div>
      </Row>
      {/**
       * Display the footer with the Pause button. BiteAcquisition has no resume
       * button, because the selected food mask may no longer be usable (e.g.,
       * if the robot moved the food items)
       */}
      <Footer
        pauseCallback={() => console.log('Pause Clicked')}
        backCallback={backCallback}
        backMealState={backMealState}
        resumeCallback={null}
        paused={paused}
        setPaused={setPaused}
      />
    </>
  )
}
BiteAcquisition.propTypes = {
  debug: PropTypes.bool.isRequired
}

export default BiteAcquisition
