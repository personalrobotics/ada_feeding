// React Imports
import React from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'

/**
 * The BiteDone component appears after the robot has moved to the user's mouth,
 * and waits for the user to specify that they have finished the bite before
 * moving back to the staging area.
 */
const BiteDone = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)

  /**
   * Callback function for when the user is done with their bite.
   */
  function doneWithBite() {
    console.log('readyForBite')
    setMealState(MEAL_STATE.R_MovingAbovePlate)
  }

  /**
   * Callback function for if the user decides to cancel the bite.
   *
   * TODO: Think more carefully about what cancelBite at this stage should do!
   * Maybe replace with a more descriptive button (e.g., "return to staging.")
   */
  function cancelBite() {
    console.log('cancelBite')
    setMealState(MEAL_STATE.R_MovingToStagingLocation)
  }

  // Render the component
  return (
    <div style={{ display: 'block', width: '100%', height: '115vh', overflowX: 'hidden', overflowY: 'auto' }} className='outer'>
      {/* Give the user the option to cancel this bite */}
      <div style={{ display: 'inline' }}>
        <Button className='cancelButton' style={{ fontSize: '24px' }} onClick={cancelBite}>
          🗑 Cancel Bite
        </Button>
      </div>

      {/* Ask the user whether they're ready for the bite */}
      <p className='transitionMessage' style={{ marginBottom: '0px' }}>
        Click the below button to indicate the completion of your bite.
      </p>
      <Row className='justify-content-center mx-auto my-2 w-75'>
        <Button
          variant='primary'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={doneWithBite}
          style={{ width: '75%', fontSize: '35px' }}
        >
          Done with Bite
        </Button>
      </Row>
    </div>
  )
}

export default BiteDone
