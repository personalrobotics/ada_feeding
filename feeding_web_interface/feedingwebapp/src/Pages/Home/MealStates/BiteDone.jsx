/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React Imports
import React, { useCallback, useEffect, useRef, useState } from 'react'
import Button from 'react-bootstrap/Button'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'

// Local Imports
import { useROS, createROSService, createROSServiceRequest, subscribeToROSTopic, unsubscribeFromROSTopic } from '../../../ros/ros_helpers'
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import { FOOD_ON_FORK_DETECTION_TOPIC, FOOD_ON_FORK_DETECTION_TOPIC_MSG, ROS_SERVICE_NAMES, MOVING_STATE_ICON_DICT } from '../../Constants'

/**
 * The BiteDone component appears after the robot has moved to the user's mouth,
 * and waits for the user to specify that they have finished the bite before
 * moving back to above plate.
 */
const BiteDone = () => {
  // Store the remaining time before auto-continuing
  const [remainingSeconds, setRemainingSeconds] = useState(null)
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  const setInNonMovingState = useGlobalState((state) => state.setInNonMovingState)
  const biteDoneAutoContinue = useGlobalState((state) => state.biteDoneAutoContinue)
  const setBiteDoneAutoContinue = useGlobalState((state) => state.setBiteDoneAutoContinue)
  const biteDoneAutoContinueSecs = useGlobalState((state) => state.biteDoneAutoContinueSecs)
  const biteDoneAutoContinueProbThresh = useGlobalState((state) => state.biteDoneAutoContinueProbThresh)
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
  let textFontSize = isPortrait ? 3 : 2.5
  let sizeSuffix = isPortrait ? 'vh' : 'vw'
  let buttonWidth = isPortrait ? '30vh' : '30vw'
  let buttonHeight = isPortrait ? '16vh' : '16vw'
  let iconWidth = isPortrait ? '28vh' : '28vw'
  let iconHeight = isPortrait ? '14vh' : '14vw'

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

  /*
   * Create refs to store the interval for the food-on-fork detection timers.
   * Note we need two timers, because the first timer set's remainingTime, whereas
   * we can't set remainingTime in the second timer otherwise it will attempt to
   * set state on an unmounted component.
   **/
  const timerRef = useRef(null)
  const finalTimerRef = useRef(null)
  const clearTimer = useCallback(() => {
    console.log('Clearing timer')
    if (finalTimerRef.current) {
      clearInterval(finalTimerRef.current)
      finalTimerRef.current = null
    }
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
      setRemainingSeconds(null)
    }
  }, [setRemainingSeconds, timerRef])

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * When the component is first mounted, and any time auto-continue changes,
   * if auto-continue is not enabled, set it to a non-moving state. We don't need
   * to reset it when the component is un-mounted since it will get set when
   * the mealState is updated.
   */
  useEffect(() => {
    if (biteDoneAutoContinue) {
      setInNonMovingState(false)
    } else {
      setInNonMovingState(true)
    }
  }, [biteDoneAutoContinue, setInNonMovingState])

  /**
   * Subscribe to the ROS Topic with the food-on-fork detection result. This is
   * created in local state to avoid re-creating it upon every re-render.
   */
  const foodOnForkDetectionCallback = useCallback(
    (message) => {
      console.log('Got food-on-fork detection message', message)
      if (biteDoneAutoContinue && message.status === 1) {
        if (message.probability < biteDoneAutoContinueProbThresh) {
          console.log('Auto-continuing in ', remainingSeconds, ' seconds')
          // Don't override an existing timer
          if (!timerRef.current) {
            setRemainingSeconds(biteDoneAutoContinueSecs)
            timerRef.current = setInterval(() => {
              setRemainingSeconds((prev) => {
                if (prev <= 1) {
                  clearTimer()
                  // In the remaining time, move above plate
                  finalTimerRef.current = setInterval(() => {
                    clearInterval(finalTimerRef.current)
                    moveAbovePlate()
                  }, (prev - 1) * 1000)
                  return null
                } else {
                  return prev - 1
                }
              })
            }, 1000)
          }
        } else {
          console.log('Not auto-continuing')
          clearTimer()
        }
      }
    },
    [
      biteDoneAutoContinue,
      biteDoneAutoContinueProbThresh,
      biteDoneAutoContinueSecs,
      finalTimerRef,
      remainingSeconds,
      clearTimer,
      moveAbovePlate,
      setRemainingSeconds,
      timerRef
    ]
  )
  useEffect(() => {
    let topic = subscribeToROSTopic(
      ros.current,
      FOOD_ON_FORK_DETECTION_TOPIC,
      FOOD_ON_FORK_DETECTION_TOPIC_MSG,
      foodOnForkDetectionCallback
    )
    /**
     * In practice, because the values passed in in the second argument of
     * useEffect will not change on re-renders, this return statement will
     * only be called when the component unmounts.
     */
    return () => {
      unsubscribeFromROSTopic(topic, foodOnForkDetectionCallback)
    }
  }, [foodOnForkDetectionCallback])

  /**
   * Create the ROS Service. This is created in local state to avoid re-creating
   * it upon every re-render.
   */
  let { serviceName, messageType } = ROS_SERVICE_NAMES[MEAL_STATE.U_BiteDone]
  let toggleFoodOnForkDetectionService = useRef(createROSService(ros.current, serviceName, messageType))

  /**
   * Toggles food-on-fork detection on the first time this component is rendered,
   * but not upon additional re-renders. See here for more details on how `useEffect`
   * achieves this goal: https://stackoverflow.com/a/69264685
   */
  useEffect(() => {
    // Create a service request
    let request = createROSServiceRequest({ data: true })
    // Call the service
    let service = toggleFoodOnForkDetectionService.current
    service.callService(request, (response) => console.log('Got toggle food on fork detection service response', response))

    /**
     * In practice, because the values passed in in the second argument of
     * useEffect will not change on re-renders, this return statement will
     * only be called when the component unmounts.
     */
    return () => {
      // Create a service request
      let request = createROSServiceRequest({ data: false })
      // Call the service
      service.callService(request, (response) => console.log('Got toggle food on fork detection service response', response))
    }
  }, [toggleFoodOnForkDetectionService])

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
              name='biteDoneAutoContinue'
              type='checkbox'
              checked={biteDoneAutoContinue}
              onChange={(e) => {
                clearTimer()
                setBiteDoneAutoContinue(e.target.checked)
              }}
              style={{ transform: 'scale(2.0)', verticalAlign: 'middle', marginRight: '15px' }}
            />
            Auto-continue
          </p>
        </View>
        <View
          style={{
            flex: 1,
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize.toString() + sizeSuffix }}>
            {biteDoneAutoContinue
              ? remainingSeconds === null
                ? 'Waiting for the fork to be empty'
                : 'Moving away in ' + remainingSeconds + ' secs'
              : ''}
          </p>
        </View>
        <View style={{ flex: 9, flexDirection: dimension, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
            {/* Ask the user whether they want to move to above plate position */}
            <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize.toString() + sizeSuffix }}>
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
            <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize.toString() + sizeSuffix }}>
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
            <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize.toString() + sizeSuffix }}>
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
      </>
    )
  }, [
    biteDoneAutoContinue,
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
    remainingSeconds,
    clearTimer,
    setBiteDoneAutoContinue,
    textFontSize,
    sizeSuffix
  ])

  // Render the component
  return <>{fullPageView()}</>
}

export default BiteDone
