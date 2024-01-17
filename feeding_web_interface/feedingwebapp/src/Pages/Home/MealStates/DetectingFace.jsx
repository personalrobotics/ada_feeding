// React Imports
import React, { useCallback, useState } from 'react'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'
import Button from 'react-bootstrap/Button'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import { MOVING_STATE_ICON_DICT } from '../../Constants'
import DetectingFaceSubcomponent from './DetectingFaceSubcomponent'

/**
 * The DetectingFace component appears after the robot has moved to the staging
 * configuration. It displays the output of face detection, and automatically
 * moves on to `R_MovingToMouth` when a face is detected.
 */
const DetectingFace = (props) => {
  // Keep track of whether a mouth has been detected or not
  const [mouthDetected, setMouthDetected] = useState(false)
  // Get the relevant global variables
  const prevMealState = useGlobalState((state) => state.prevMealState)
  const setMealState = useGlobalState((state) => state.setMealState)
  const setMoveToMouthActionGoal = useGlobalState((state) => state.setMoveToMouthActionGoal)
  const faceDetectionAutoContinue = useGlobalState((state) => state.faceDetectionAutoContinue)
  const setFaceDetectionAutoContinue = useGlobalState((state) => state.setFaceDetectionAutoContinue)
  // Get icon image for move to mouth
  let moveToMouthImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth]
  let moveToRestingImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToRestingPosition]
  let moveAbovePlateImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate]
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'
  let otherDimension = isPortrait ? 'row' : 'column'
  // Font size for text
  let textFontSize = 3
  let buttonWidth = 30
  let buttonHeight = 18
  let iconWidth = 28
  let iconHeight = 16
  let sizeSuffix = isPortrait ? 'vh' : 'vw'

  /**
   * Callback function for proceeding to move to the mouth position.
   */
  const moveToMouthCallback = useCallback(() => {
    console.log('Transitioning to R_MovingToMouth')
    setMealState(MEAL_STATE.R_MovingToMouth)
    // Set mouth detected to false for the next time this screen comes up
    setMouthDetected(false)
  }, [setMealState, setMouthDetected])

  /**
   * Callback function for proceeding to move to the resting position.
   */
  const moveToRestingCallback = useCallback(() => {
    console.log('Transitioning to R_MovingToRestingPosition')
    setMealState(MEAL_STATE.R_MovingToRestingPosition)
    // Set mouth detected to false for the next time this screen comes up
    setMouthDetected(false)
  }, [setMealState, setMouthDetected])

  /**
   * Callback function for proceeding to move above the plate.
   */
  const moveAbovePlateCallback = useCallback(() => {
    console.log('Transitioning to R_MovingAbovePlate')
    setMealState(MEAL_STATE.R_MovingAbovePlate)
    // Set mouth detected to false for the next time this screen comes up
    setMouthDetected(false)
  }, [setMealState, setMouthDetected])

  /**
   * Callback for when a face is detected within the correct range.
   */
  const faceDetectedCallback = useCallback(
    (message) => {
      console.log('Face detected callback')
      setMouthDetected(true)
      setMoveToMouthActionGoal({
        face_detection: message
      })
      // Automatically move on to the next stage if a face is detected
      // If the app got to this screen after moving away from the user's mouth,
      // don't auto-continue. Only do so if it gets to this page from
      // R_MovingToStagingConfiguration
      if (faceDetectionAutoContinue && prevMealState !== MEAL_STATE.R_MovingFromMouth) {
        moveToMouthCallback()
      }
    },
    [faceDetectionAutoContinue, moveToMouthCallback, prevMealState, setMoveToMouthActionGoal]
  )

  /** Get the full page view
   *
   * @returns {JSX.Element} the the full page view
   */
  const fullPageView = useCallback(() => {
    return (
      <>
        <View
          style={{
            flex: 1,
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%'
          }}
        >
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize.toString() + sizeSuffix }}>
            <input
              name='faceDetectionAutoContinue'
              type='checkbox'
              checked={faceDetectionAutoContinue}
              onChange={(e) => setFaceDetectionAutoContinue(e.target.checked)}
              style={{ transform: 'scale(2.0)', verticalAlign: 'middle', marginRight: '15px' }}
            />
            Auto-continue
          </p>
        </View>
        <View
          style={{
            flex: 9,
            flexDirection: dimension,
            alignItems: 'center',
            width: '100%'
          }}
        >
          <View style={{ flex: 5, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
            <DetectingFaceSubcomponent faceDetectedCallback={faceDetectedCallback} webrtcURL={props.webrtcURL} />
          </View>
          <View style={{ flex: 3, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
            <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize.toString() + sizeSuffix }}>
              {mouthDetected ? 'Continue' : 'Continue without detection'}
            </p>
            {/* Icon to move to mouth position */}
            <Button
              variant='success'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              onClick={moveToMouthCallback}
              style={{
                width: buttonWidth.toString() + sizeSuffix,
                height: buttonHeight.toString() + sizeSuffix,
                display: 'flex',
                justifyContent: 'center',
                alignContent: 'center'
              }}
            >
              <img
                src={moveToMouthImage}
                alt='move_to_mouth_image'
                className='center'
                style={{ width: iconWidth.toString() + sizeSuffix, height: iconHeight.toString() + sizeSuffix }}
              />
            </Button>
          </View>
          <View
            style={{
              flex: 2,
              flexDirection: otherDimension,
              alignItems: 'center',
              justifyContent: 'center',
              width: '100%',
              height: '100%'
            }}
          >
            <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
              <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: (textFontSize * 0.66).toString() + sizeSuffix }}>
                Move to Resting
              </p>
              {/* Icon to move to mouth position */}
              <Button
                variant='warning'
                className='mx-2 mb-2 btn-huge'
                size='lg'
                onClick={moveToRestingCallback}
                style={{
                  width: (buttonWidth / 2).toString() + sizeSuffix,
                  height: (buttonHeight / 2).toString() + sizeSuffix,
                  display: 'flex',
                  justifyContent: 'center',
                  alignContent: 'center'
                }}
              >
                <img
                  src={moveToRestingImage}
                  alt='move_to_resting_image'
                  className='center'
                  style={{ width: (iconWidth / 2).toString() + sizeSuffix, height: (iconHeight / 2).toString() + sizeSuffix }}
                />
              </Button>
            </View>
            <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
              <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: (textFontSize * 0.66).toString() + sizeSuffix }}>
                Move Above Plate
              </p>
              {/* Icon to move to mouth position */}
              <Button
                variant='warning'
                className='mx-2 mb-2 btn-huge'
                size='lg'
                onClick={moveAbovePlateCallback}
                style={{
                  width: (buttonWidth / 2).toString() + sizeSuffix,
                  height: (buttonHeight / 2).toString() + sizeSuffix,
                  display: 'flex',
                  justifyContent: 'center',
                  alignContent: 'center'
                }}
              >
                <img
                  src={moveAbovePlateImage}
                  alt='move_above_plate_image'
                  className='center'
                  style={{ width: (iconWidth / 2).toString() + sizeSuffix, height: (iconHeight / 2).toString() + sizeSuffix }}
                />
              </Button>
            </View>
          </View>
        </View>
      </>
    )
  }, [
    dimension,
    otherDimension,
    mouthDetected,
    faceDetectedCallback,
    moveToMouthCallback,
    moveToRestingCallback,
    moveAbovePlateCallback,
    moveToMouthImage,
    moveToRestingImage,
    moveAbovePlateImage,
    props.webrtcURL,
    textFontSize,
    buttonHeight,
    buttonWidth,
    sizeSuffix,
    iconHeight,
    iconWidth,
    faceDetectionAutoContinue,
    setFaceDetectionAutoContinue
  ])

  // Render the component
  return fullPageView()
}
DetectingFace.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default DetectingFace
