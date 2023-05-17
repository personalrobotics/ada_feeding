// React Imports
import React, { useCallback } from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'

/**
 * The PostMeal component appearsto the end of a meal, and lets the robot return
 * to the main menu.
 */
const PostMeal = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)

  /**
   * Callback function for when the user decides to return to the start of the app.
   */
  const returnToStartClicked = useCallback(() => {
    console.log('startFeedingClicked')
    setMealState(MEAL_STATE.U_PreMeal)
  }, [setMealState])

  // Render the component
  return (
    <Row xs={1} md={1} className='justify-content-center mx-2 my-2'>
      <p className='transitionMessage' style={{ marginBottom: '10px', marginTop: '0px', fontSize: '24px' }}>
        That was a delicious meal 😋 ! Cheers 🥂 to your good health!
      </p>
      <Button
        variant='primary'
        size='lg'
        className='btn-huge'
        id='#returnToStartBtn'
        onClick={returnToStartClicked}
        style={{ width: '75%', fontSize: '35px', marginTop: '30px' }}
      >
        Return to Start
      </Button>
    </Row>
  )
}

export default PostMeal
