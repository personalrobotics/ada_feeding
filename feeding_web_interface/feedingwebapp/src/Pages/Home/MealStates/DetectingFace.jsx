/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React Imports
import React, { useCallback, useEffect, useState } from 'react'
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
  const mealState = useGlobalState((state) => state.mealState)
  const prevMealState = useGlobalState((state) => state.prevMealState)
  const setInNonMovingState = useGlobalState((state) => state.setInNonMovingState)
  const setMealState = useGlobalState((state) => state.setMealState)
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
  // let buttonWidth = 22
  let buttonHeight = 12
  let iconWidth = 20
  let iconHeight = 10
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
   * Auto-continue is enabled if the face detection auto-continue is checked
   * and the previous meal state was R_MovingToStagingConfiguration (i.e., this
   * state is arrived to nominally).
   */
  const autoContinueIsEnabled = useCallback(() => {
    console.log('Checking auto-continue', faceDetectionAutoContinue, prevMealState)
    return faceDetectionAutoContinue && prevMealState === MEAL_STATE.R_MovingToStagingConfiguration
  }, [faceDetectionAutoContinue, prevMealState])

  /**
   * When the component is first mounted, and any time auto-continue changes,
   * if auto-continue is not enabled, set it to a non-moving state. We don't need
   * to reset it when the component is un-mounted since it will get set when
   * the mealState is updated.
   */
  useEffect(() => {
    // mealState must be included to have this re-run when we return to the
    // component even if it wasn't re-mounted.
    if (autoContinueIsEnabled()) {
      setInNonMovingState(false)
    } else {
      setInNonMovingState(true)
    }
  }, [autoContinueIsEnabled, mealState, setInNonMovingState])

  /**
   * Callback for when a face is detected within the correct range.
   */
  const faceDetectedCallback = useCallback(() => {
    setMouthDetected(true)
    // If auto-continue is enabled, move to the mouth position
    if (autoContinueIsEnabled()) {
      moveToMouthCallback()
    }
  }, [autoContinueIsEnabled, moveToMouthCallback])

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
          <View style={{ flex: 6, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
            <DetectingFaceSubcomponent faceDetectedCallback={faceDetectedCallback} webrtcURL={props.webrtcURL} />
          </View>
          <View style={{ flex: 2, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
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
                width: '90%',
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
                  width: '90%',
                  height: ((buttonHeight * 2) / 3).toString() + sizeSuffix,
                  display: 'flex',
                  justifyContent: 'center',
                  alignContent: 'center'
                }}
              >
                <img
                  src={moveToRestingImage}
                  alt='move_to_resting_image'
                  className='center'
                  style={{ width: (iconWidth / 2).toString() + sizeSuffix, height: ((iconHeight * 2) / 3).toString() + sizeSuffix }}
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
                  width: '90%',
                  height: ((buttonHeight * 2) / 3).toString() + sizeSuffix,
                  display: 'flex',
                  justifyContent: 'center',
                  alignContent: 'center'
                }}
              >
                <img
                  src={moveAbovePlateImage}
                  alt='move_above_plate_image'
                  className='center'
                  style={{ width: (iconWidth / 2).toString() + sizeSuffix, height: ((iconHeight * 2) / 3).toString() + sizeSuffix }}
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
