/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

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
  // Font size for text
  let textFontSize = '4.2vh'
  // Width for button
  let buttonWidth = '70vw'
  // Height for button
  let buttonHeight = '10vh'
  // Margin
  let margin = '5vh'

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
      <p className='transitionMessage' style={{ marginBottom: margin, marginTop: '0px', fontSize: textFontSize }}>
        That was a delicious meal 😋 ! Cheers 🥂 to your good health!
      </p>
      <Button
        variant='primary'
        size='lg'
        className='btn-huge'
        id='#returnToStartBtn'
        onClick={returnToStartClicked}
        style={{ width: buttonWidth, height: buttonHeight, fontSize: textFontSize, marginTop: margin }}
      >
        Return to Start
      </Button>
    </Row>
  )
}

export default PostMeal
