// React Imports
import React, { useCallback } from 'react'
import Button from 'react-bootstrap/Button'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import { MOVING_STATE_ICON_DICT } from '../../Constants'

/**
 * The BiteDone component appears after the robot has moved to the user's mouth,
 * and waits for the user to specify that they have finished the bite before
 * moving back to above plate.
 */
const BiteDone = () => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  // Get icon image for move above plate
  let moveAbovePlateImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate]
  // Get icon image for move to resting position
  let moveToRestingPositionImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToRestingPosition]
  // Get icom image for move to staging configuration
  let moveToStagingConfigurationImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToStagingConfiguration]
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'
  // Font size for text
  let textFontSize = isPortrait ? '3vh' : '2.5vw'
  let buttonWidth = isPortrait ? '30vh' : '30vw'
  let buttonHeight = isPortrait ? '20vh' : '20vw'
  let iconWidth = isPortrait ? '28vh' : '28vw'
  let iconHeight = isPortrait ? '18vh' : '18vw'

  /**
   * Callback function for when the user wants to move above plate.
   */
  const moveAbovePlate = useCallback(() => {
    setMealState(MEAL_STATE.R_MovingFromMouth, MEAL_STATE.R_MovingAbovePlate)
  }, [setMealState])

  /**
   * Callback function for when the user wants to move to resting position.
   */
  const moveToRestingPosition = useCallback(() => {
    setMealState(MEAL_STATE.R_MovingFromMouth, MEAL_STATE.R_MovingToRestingPosition)
  }, [setMealState])

  /**
   * Callback function for when the user wants to move to the staging configuration.
   */
  const moveToStagingConfiguration = useCallback(() => {
    setMealState(MEAL_STATE.R_MovingFromMouth, MEAL_STATE.R_DetectingFace)
  }, [setMealState])

  /** Get the full page view
   *
   * @returns {JSX.Element} the the full page view
   */
  const fullPageView = useCallback(() => {
    return (
      <View style={{ flex: 'auto', flexDirection: dimension, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          {/* Ask the user whether they want to move to above plate position */}
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            Move above plate
          </p>
          {/* Icon to move above plate */}
          <Button
            variant='success'
            className='mx-2 mb-2 btn-huge'
            size='lg'
            onClick={moveAbovePlate}
            style={{ width: buttonWidth, height: buttonHeight }}
          >
            <img
              src={moveAbovePlateImage}
              alt='move_above_plate_image'
              className='center'
              style={{ width: iconWidth, height: iconHeight }}
            />
          </Button>
        </View>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          {/* Ask the user whether they want to move to resting position */}
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            Rest to the side
          </p>
          {/* Icon to move to resting position */}
          <Button
            variant='warning'
            className='mx-2 mb-2 btn-huge'
            size='lg'
            onClick={moveToRestingPosition}
            style={{ width: buttonWidth, height: buttonHeight }}
          >
            <img
              src={moveToRestingPositionImage}
              alt='move_to_resting_image'
              className='center'
              style={{ width: iconWidth, height: iconHeight }}
            />
          </Button>
        </View>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          {/* Ask the user whether they want to move to resting position */}
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            Move away from mouth
          </p>
          {/* Icon to move to resting position */}
          <Button
            variant='warning'
            className='mx-2 mb-2 btn-huge'
            size='lg'
            onClick={moveToStagingConfiguration}
            style={{ width: buttonWidth, height: buttonHeight }}
          >
            <img
              src={moveToStagingConfigurationImage}
              alt='move_to_staging_image'
              className='center'
              style={{ width: iconWidth, height: iconHeight }}
            />
          </Button>
        </View>
      </View>
    )
  }, [
    buttonHeight,
    buttonWidth,
    dimension,
    iconHeight,
    iconWidth,
    moveAbovePlate,
    moveAbovePlateImage,
    moveToRestingPosition,
    moveToRestingPositionImage,
    moveToStagingConfiguration,
    moveToStagingConfigurationImage,
    textFontSize
  ])

  // Render the component
  return <>{fullPageView()}</>
}

export default BiteDone
