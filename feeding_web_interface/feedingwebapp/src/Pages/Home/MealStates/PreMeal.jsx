// React Imports
import React from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'

/**
 * The PreMeal component appears before the meal starts, and gives the user a
 * Start Feeding button to initiate feeding. Further, PreMeal is the only meal
 * state as of now where users can modify the settings.
 *
 * @params {object} props - contains any properties passed to this Component
 */
const PreMeal = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)

  /**
   * Callback function for when the user decides to start feeding using the app.
   */
  function startFeedingClicked() {
    console.log('startFeedingClicked')
    setMealState(MEAL_STATE.R_MovingAbovePlate)
  }

  // Render the component
  return (
    <Row xs={1} md={1} className='justify-content-center mx-2 my-2'>
      <p className='transitionMessage' style={{ marginBottom: '10px', marginTop: '0px', fontSize: '24px' }}>
        Hello!👋 I am ADA&apos;s faithful assistant, ADAWebapp! Bon Appétit! 😋
      </p>
      <Button
        variant='primary'
        size='lg'
        className='btn-huge'
        id='#startFeedingBtn'
        onClick={startFeedingClicked}
        style={{ width: '75%', fontSize: '35px', marginTop: '30px' }}
      >
        Start Feeding
      </Button>
    </Row>
  )
}

export default PreMeal
