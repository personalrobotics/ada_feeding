// React Imports
import React, { useCallback, useEffect, useState, useRef } from 'react'
import Button from 'react-bootstrap/Button'
import { useMediaQuery } from 'react-responsive'
import { toast } from 'react-toastify'
import { View } from 'react-native'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import { MOVING_STATE_ICON_DICT } from '../../Constants'
import { useROS, createROSService, createROSServiceRequest, subscribeToROSTopic, unsubscribeFromROSTopic } from '../../../ros/ros_helpers'
import {
  ACQUISITION_REPORT_SERVICE_NAME,
  ACQUISITION_REPORT_SERVICE_TYPE,
  FOOD_ON_FORK_DETECTION_TOPIC,
  FOOD_ON_FORK_DETECTION_TOPIC_MSG,
  REGULAR_CONTAINER_ID,
  ROS_SERVICE_NAMES
} from '../../Constants'

/**
 * The BiteAcquisitionCheck component appears after the robot has attempted to
 * acquire a bite, and asks the user whether it succeeded at acquiring the bite.
 */
const BiteAcquisitionCheck = () => {
  // Store the remining time before auto-continuing
  const [remainingSeconds, setRemainingSeconds] = useState(null)
  // Get the relevant global variables
  const prevMealState = useGlobalState((state) => state.prevMealState)
  const setInNonMovingState = useGlobalState((state) => state.setInNonMovingState)
  const setMealState = useGlobalState((state) => state.setMealState)
  const biteAcquisitionCheckAutoContinue = useGlobalState((state) => state.biteAcquisitionCheckAutoContinue)
  const setBiteAcquisitionCheckAutoContinue = useGlobalState((state) => state.setBiteAcquisitionCheckAutoContinue)
  const biteAcquisitionCheckAutoContinueSecs = useGlobalState((state) => state.biteAcquisitionCheckAutoContinueSecs)
  const biteAcquisitionCheckAutoContinueProbThreshLower = useGlobalState((state) => state.biteAcquisitionCheckAutoContinueProbThreshLower)
  const biteAcquisitionCheckAutoContinueProbThreshUpper = useGlobalState((state) => state.biteAcquisitionCheckAutoContinueProbThreshUpper)
  // Get icon image for move above plate
  let moveAbovePlateImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate]
  // Get icon image for move to mouth
  let moveToStagingConfigurationImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToStagingConfiguration]
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'
  // Font size for text
  let textFontSize = isPortrait ? '3vh' : '3vw'
  let buttonWidth = isPortrait ? '30vh' : '30vw'
  let buttonHeight = isPortrait ? '20vh' : '20vw'
  let iconWidth = isPortrait ? '28vh' : '28vw'
  let iconHeight = isPortrait ? '18vh' : '18vw'

  // Configure AcquisitionReport service
  const lastMotionActionResponse = useGlobalState((state) => state.lastMotionActionResponse)
  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)
  /**
   * Create the ROS Service Client for reporting success/failure
   */
  let acquisitionReportService = useRef(createROSService(ros.current, ACQUISITION_REPORT_SERVICE_NAME, ACQUISITION_REPORT_SERVICE_TYPE))
  /**
   * Create the ROS Service. This is created in local state to avoid re-creating
   * it upon every re-render.
   */
  let { serviceName, messageType } = ROS_SERVICE_NAMES[MEAL_STATE.U_BiteAcquisitionCheck]
  let toggleFoodOnForkDetectionService = useRef(createROSService(ros.current, serviceName, messageType))

  /**
   * Callback function for when the user indicates that the bite acquisition
   * succeeded.
   */
  const acquisitionSuccess = useCallback(() => {
    console.log('acquisitionSuccess')
    // NOTE: This uses the ToastContainer in Header
    toast.info('Reporting Food Acquisition Success!', {
      containerId: REGULAR_CONTAINER_ID,
      toastId: 'foodAcquisitionSuccess'
    })
    // Create a service request
    let request = createROSServiceRequest({
      loss: 0.0,
      action_index: lastMotionActionResponse.action_index,
      posthoc: lastMotionActionResponse.posthoc,
      id: lastMotionActionResponse.selection_id
    })
    // Call the service
    let service = acquisitionReportService.current
    service.callService(request, (response) => console.log('Got acquisition report response', response))
    setMealState(MEAL_STATE.R_MovingToStagingConfiguration)
  }, [lastMotionActionResponse, setMealState])

  /**
   * Callback function for when the user indicates that the bite acquisition
   * failed.
   */
  const acquisitionFailure = useCallback(() => {
    console.log('acquisitionFailure')
    // NOTE: This uses the ToastContainer in Header
    toast.info('Reporting Food Acquisition Failure.', {
      containerId: REGULAR_CONTAINER_ID,
      toastId: 'foodAcquisitionFailure'
    })
    // Create a service request
    let request = createROSServiceRequest({
      loss: 1.0,
      action_index: lastMotionActionResponse.action_index,
      posthoc: lastMotionActionResponse.posthoc,
      id: lastMotionActionResponse.selection_id
    })
    // Call the service
    let service = acquisitionReportService.current
    service.callService(request, (response) => console.log('Got acquisition report response', response))
    setMealState(MEAL_STATE.R_MovingAbovePlate)
  }, [lastMotionActionResponse, setMealState])

  /*
   * Create refs to store the interval for the food-on-fork detection timers.
   * Note we need two timers, because the first timer set's remainingTime, whereas
   * we can't set remainingTime in the second timer otherwise it will attempt to
   * set state on an unmounted component.
   **/
  const timerWasForFof = useRef(null)
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
      timerWasForFof.current = null
    }
  }, [setRemainingSeconds, timerRef, timerWasForFof, finalTimerRef])

  /**
   * Auto-continue is enabled if the bite acquisition check auto-continue is checked
   * and the previous meal state was R_BiteAcquisition (i.e., this
   * state is arrived to nominally).
   */
  const autoContinueIsEnabled = useCallback(() => {
    console.log('Checking auto-continue', biteAcquisitionCheckAutoContinue, prevMealState)
    return biteAcquisitionCheckAutoContinue && prevMealState === MEAL_STATE.R_BiteAcquisition
  }, [biteAcquisitionCheckAutoContinue, prevMealState])

  /**
   * When the component is first mounted, and any time auto-continue changes,
   * if auto-continue is not enabled, set it to a non-moving state. We don't need
   * to reset it when the component is un-mounted since it will get set when
   * the mealState is updated.
   */
  useEffect(() => {
    if (autoContinueIsEnabled()) {
      setInNonMovingState(false)
    } else {
      setInNonMovingState(true)
    }
  }, [autoContinueIsEnabled, setInNonMovingState])

  /**
   * Subscribe to the ROS Topic with the food-on-fork detection result. This is
   * created in local state to avoid re-creating it upon every re-render.
   */
  const foodOnForkDetectionCallback = useCallback(
    (message) => {
      console.log('Got food-on-fork detection message', message)
      // Only auto-continue if the previous state was Bite Acquisition
      if (autoContinueIsEnabled()) {
        let callbackFn = null
        if (message.status === -1 || (message.status === 1 && message.probability < biteAcquisitionCheckAutoContinueProbThreshLower)) {
          console.log('No FoF. Auto-continuing in ', remainingSeconds, ' seconds')
          if (timerWasForFof.current === true) {
            clearTimer()
          }
          timerWasForFof.current = false
          callbackFn = acquisitionFailure
        } else if (message.status === 1 && message.probability > biteAcquisitionCheckAutoContinueProbThreshUpper) {
          console.log('FoF. Auto-continuing in ', remainingSeconds, ' seconds')
          if (timerWasForFof.current === false) {
            clearTimer()
          }
          timerWasForFof.current = true
          callbackFn = acquisitionSuccess
        } else {
          console.log('Not auto-continuing due to probability between thresholds or errors')
          clearTimer()
        }
        // Don't override an existing timer
        if (!timerRef.current && callbackFn !== null) {
          setRemainingSeconds(biteAcquisitionCheckAutoContinueSecs)
          timerRef.current = setInterval(() => {
            setRemainingSeconds((prev) => {
              if (prev <= 1) {
                clearTimer()
                // In the remaining time, move above plate
                finalTimerRef.current = setInterval(() => {
                  clearInterval(finalTimerRef.current)
                  callbackFn()
                }, (prev - 1) * 1000)
                return null
              } else {
                return prev - 1
              }
            })
          }, 1000)
        }
      }
    },
    [
      acquisitionSuccess,
      acquisitionFailure,
      autoContinueIsEnabled,
      biteAcquisitionCheckAutoContinueProbThreshLower,
      biteAcquisitionCheckAutoContinueProbThreshUpper,
      biteAcquisitionCheckAutoContinueSecs,
      finalTimerRef,
      remainingSeconds,
      clearTimer,
      setRemainingSeconds,
      timerRef,
      timerWasForFof
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

  /**
   * Get the ready for bite text to render.
   *
   * @returns {JSX.Element} the ready for bite text
   */
  const readyForBiteText = useCallback(() => {
    return (
      <>
        {/* Ask the user whether they want to move to mouth position */}
        <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
          Ready for bite? Move to mouth.
        </p>
      </>
    )
  }, [textFontSize])

  /**
   * Get the ready for bite button to render.
   *
   * @returns {JSX.Element} the ready for bite button
   */
  const readyForBiteButton = useCallback(() => {
    return (
      <>
        {/* Icon to move to mouth position */}
        <Button
          variant='success'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={acquisitionSuccess}
          style={{ width: buttonWidth, height: buttonHeight }}
        >
          <img
            src={moveToStagingConfigurationImage}
            alt='move_to_mouth_image'
            className='center'
            style={{ width: iconWidth, height: iconHeight }}
          />
        </Button>
      </>
    )
  }, [moveToStagingConfigurationImage, acquisitionSuccess, buttonHeight, buttonWidth, iconHeight, iconWidth])

  /**
   * Get the re-acquire bite text to render.
   *
   * @returns {JSX.Element} the re-acquire bite text
   */
  const reacquireBiteText = useCallback(() => {
    return (
      <>
        {/* Ask the user whether they want to try acquiring bite again */}
        <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
          Re-acquire bite? Move above plate.
        </p>
      </>
    )
  }, [textFontSize])

  /**
   * Get the re-acquire bite button to render.
   *
   * @returns {JSX.Element} the re-acquire bite button
   */
  const reacquireBiteButton = useCallback(() => {
    return (
      <>
        {/* Icon for move above plate */}
        <Button
          variant='warning'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          onClick={acquisitionFailure}
          style={{ width: buttonWidth, height: buttonHeight }}
        >
          <img src={moveAbovePlateImage} alt='move_above_plate_image' className='center' style={{ width: iconWidth, height: iconHeight }} />
        </Button>
      </>
    )
  }, [acquisitionFailure, moveAbovePlateImage, buttonHeight, buttonWidth, iconHeight, iconWidth])

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
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            <input
              name='biteAcquisitionCheckAutoContinue'
              type='checkbox'
              checked={biteAcquisitionCheckAutoContinue}
              onChange={(e) => {
                clearTimer()
                setBiteAcquisitionCheckAutoContinue(e.target.checked)
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
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            {autoContinueIsEnabled()
              ? remainingSeconds === null
                ? 'Checking for food on fork...'
                : timerWasForFof.current
                ? 'Moving to your face in ' + remainingSeconds + ' secs'
                : 'Moving above plate in ' + remainingSeconds + ' secs'
              : ''}
          </p>
        </View>
        <View style={{ flex: 'auto', flexDirection: dimension, alignItems: 'center', justifyContent: 'center' }}>
          <View style={{ flex: 5, alignItems: 'center', justifyContent: 'center' }}>
            {readyForBiteText()}
            {readyForBiteButton()}
          </View>
          <View style={{ flex: 5, alignItems: 'center', justifyContent: 'center' }}>
            {reacquireBiteText()}
            {reacquireBiteButton()}
          </View>
        </View>
      </>
    )
  }, [
    autoContinueIsEnabled,
    biteAcquisitionCheckAutoContinue,
    clearTimer,
    dimension,
    readyForBiteButton,
    readyForBiteText,
    reacquireBiteButton,
    reacquireBiteText,
    remainingSeconds,
    setBiteAcquisitionCheckAutoContinue,
    textFontSize
  ])

  // Render the component
  return <>{fullPageView()}</>
}

export default BiteAcquisitionCheck
