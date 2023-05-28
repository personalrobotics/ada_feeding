// React Imports
import React, { useCallback } from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import { MOVING_STATE_ICON_DICT } from '../../Constants'

/**
 * The BiteAcquisitionCheck component appears after the robot has attempted to
 * acquire a bite, and asks the user whether it succeeded at acquiring the bite.
 */
const BiteAcquisitionCheck = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  // Get icon image for move above plate
  let moveAbovePlateImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate]
  // Get icon image for move to mouth
  let moveToMouthImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth]

  /**
   * Callback function for when the user indicates that the bite acquisition
   * succeeded.
   */
  const acquisitionSuccess = useCallback(() => {
    console.log('acquisitionSuccess')
    setMealState(MEAL_STATE.R_MovingToMouth)
  }, [setMealState])

  /**
   * Callback function for when the user indicates that the bite acquisition
   * failed.
   */
  const acquisitionFailure = useCallback(() => {
    console.log('acquisitionFailure')
    setMealState(MEAL_STATE.R_MovingAbovePlate)
  }, [setMealState])

  // Render the component
  return (
    <div style={{ display: 'block', width: '100%', height: '115vh', overflowX: 'hidden', overflowY: 'auto' }} className='outer'>
      {/* Ask the user whether they want to move to mouth position */}
      <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: '140%' }}>
        Ready for bite? Move to mouth.
      </p>
      {/* Icon to move to mouth position */}
      <Row className='justify-content-center mx-auto w-75'>
        <Button
          variant='success'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={acquisitionSuccess}
          style={{ width: '300px', height: '200px' }}
        >
          <img src={moveToMouthImage} alt='move_to_mouth_image' className='center' />
        </Button>
      </Row>
      {/* Add empty space */}
      <div className='justify-content-center mx-auto mb-1 row'>&nbsp;</div>
      <Row className='justify-content-center mx-auto mb-2'>
        {/* Ask the user whether they want to try acquiring bite again */}
        <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: '140%' }}>
          Re-acquire bite? Move above plate.
        </p>
        {/* Icon for move above plate */}
        <Button
          variant='warning'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={acquisitionFailure}
          style={{ width: '300px', height: '200px' }}
        >
          <img src={moveAbovePlateImage} alt='move_above_plate_image' className='center' />
        </Button>
      </Row>
    </div>
  )
}

export default BiteAcquisitionCheck
