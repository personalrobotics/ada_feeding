// React Imports
import React, { useCallback } from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'

/**
 * The PreMeal component appears before the meal starts, and gives the user a
 * Start Feeding button to initiate feeding. Further, PreMeal is the only meal
 * state as of now where users can modify the settings.
 */
const PreMeal = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  // Font size for text
  let textFontSize = '4.2vh'
  // Width for button
  let buttonWidth = '70vw'
  // Height for button
  let buttonHeight = '10vh'
  // Margin
  let margin = '5vh'

  /**
   * Callback function for when the user decides to start feeding using the app.
   */
  const startFeedingClicked = useCallback(() => {
    console.log('startFeedingClicked')
    setMealState(MEAL_STATE.R_MovingAbovePlate)
  }, [setMealState])

  // Render the component
  return (
    <Row xs={1} md={1} className='justify-content-center mx-2 my-2'>
      <p className='transitionMessage' style={{ marginBottom: margin, marginTop: '0', fontSize: textFontSize }}>
        Hello!👋 I am ADA&apos;s faithful assistant, ADAWebapp! Bon Appétit! 😋
      </p>
      <Button
        variant='primary'
        size='lg'
        className='btn-huge'
        id='#startFeedingBtn'
        onClick={startFeedingClicked}
        style={{ width: buttonWidth, height: buttonHeight, fontSize: textFontSize, marginTop: margin }}
      >
        Start Feeding
      </Button>
    </Row>
  )
}

export default PreMeal
