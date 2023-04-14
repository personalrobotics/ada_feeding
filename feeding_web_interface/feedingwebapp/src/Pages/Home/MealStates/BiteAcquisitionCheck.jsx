// React Imports
import React from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'

/**
 * The BiteAcquisitionCheck component appears after the robot has attempted to
 * acquire a bite, and asks the user whether it succeeded at acquiring the bite.
 */
const BiteAcquisitionCheck = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)

  /**
   * Callback function for when the user indicates that the bite acquisition
   * succeeded.
   */
  function acquisitionSuccess() {
    console.log('acquisitionSuccess')
    setMealState(MEAL_STATE.R_MovingToStagingLocation)
  }

  /**
   * Callback function for when the user indicates that the bite acquisition
   * failed.
   */
  function acquisitionFailure() {
    console.log('acquisitionFailure')
    setMealState(MEAL_STATE.R_BiteAcquisition)
  }

  // Render the component
  return (
    <div style={{ display: 'block', width: '100%', height: '115vh', overflowX: 'hidden', overflowY: 'auto' }} className='outer'>
      {/**
       * Ask the user whether the robot succeeded at picking up food or not.
       */}
      <p className='transitionMessage' style={{ marginBottom: '0px' }}>
        Is there food on the fork?
      </p>
      <Row className='justify-content-center mx-auto my-2 w-75'>
        <Button
          variant='primary'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={acquisitionSuccess}
          style={{ width: '75%', fontSize: '35px' }}
        >
          Yes, proceed
        </Button>
        <Button
          variant='primary'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={acquisitionFailure}
          style={{ width: '75%', fontSize: '35px' }}
        >
          No, try again
        </Button>
      </Row>
    </div>
  )
}

export default BiteAcquisitionCheck
