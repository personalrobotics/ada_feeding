// React Imports
import React from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'

/**
 * The BiteInitiation component appears after the robot has moved to the staging
 * position, and waits for the user to indicate that they are ready for a bite.
 */
const BiteInitiation = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)

  /**
   * Callback function for when the user is ready for their bite.
   */
  function readyForBite() {
    setMealState(MEAL_STATE.R_MovingToMouth)
  }

  /**
   * Callback function for when the user wants to move above plate.
   */
  function moveAbovePlate() {
    setMealState(MEAL_STATE.R_MovingAbovePlate)
  }

  // Render the component
  return (
    <div style={{ display: 'block', width: '100%', height: '115vh', overflowX: 'hidden', overflowY: 'auto' }} className='outer'>
      <Row className='justify-content-center mx-auto my-2'>
        {/* Ask the user whether they're ready for a bite and if they want to move to mouth position */}
        <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: '140%' }}>
          Ready for bite? Move to mouth.
        </p>
        {/* Icon to move to mouth */}
        <Button
          variant='success'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={readyForBite}
          style={{ width: '300px', height: '200px' }}
        >
          <img src='/robot_state_imgs/move_to_mouth_position.svg' alt='move_to_mouth_image' className='center' />
        </Button>
      </Row>
      {/* Add empty space */}
      <div className='justify-content-center mx-auto my-3 row'>&nbsp;</div>
      <Row className='justify-content-center mx-auto mt-5'>
        {/* Ask the user whether they want to move to above plate position */}
        <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: '140%' }}>
          Cancel bite and move above plate.
        </p>
        {/* Icon to move above plate */}
        <Button
          variant='warning'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={moveAbovePlate}
          style={{ width: '300px', height: '200px' }}
        >
          <img src='/robot_state_imgs/move_above_plate_position.svg' alt='move_above_plate_image' className='center' />
        </Button>
      </Row>
    </div>
  )
}

export default BiteInitiation
