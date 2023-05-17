// React Imports
import React, { useCallback, useEffect, useRef, useState } from 'react'
import Button from 'react-bootstrap/Button'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'
import Row from 'react-bootstrap/Row'

// Local Imports
import '../Home.css'
import { useROS, createROSService, createROSServiceRequest, subscribeToROSTopic, unsubscribeFromROSTopic } from '../../../ros/ros_helpers'
import { convertRemToPixels, scaleWidthHeightToWindow } from '../../../helpers'
import {
  FACE_DETECTION_IMG_TOPIC,
  FACE_DETECTION_TOPIC,
  FACE_DETECTION_TOPIC_MSG,
  REALSENSE_WIDTH,
  REALSENSE_HEIGHT,
  ROS_SERVICE_NAMES
} from '../../Constants'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import { FOOTER_STATE_ICON_DICT } from '../../Constants'

/**
 * The BiteInitiation component appears after the robot has moved to the staging
 * position, and waits for the user to indicate that they are ready for a bite.
 *
 * @param {boolean} debug - whether to run it in debug mode (e.g., if you aren't
 *        simulatenously running the robot) or not
 */
const BiteInitiation = (props) => {
  // Keep track of whether a mouth has been detected or not
  const [mouthDetected, setMouthDetected] = useState(false)

  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  const setDetectedMouthCenter = useGlobalState((state) => state.setDetectedMouthCenter)

  // Get icon image for move above plate
  let moveAbovePlateImage = FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate]
  // Get icon image for move to mouth position
  let moveToMouthImage = FOOTER_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth]

  /**
   * Callback function for when the user is ready for their bite.
   */
  const readyForBite = useCallback(() => {
    console.log('readyForBite')
    setMealState(MEAL_STATE.R_MovingToMouth)
  }, [setMealState])

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * Subscribe to the ROS Topic with the face detection result. This is created
   * in local state to avoid re-creating it upon every re-render.
   */
  const faceDetectionCallback = useCallback(
    (message) => {
      if (message.is_face_detected) {
        setMouthDetected(message.is_face_detected)
        setDetectedMouthCenter(message.detected_mouth_center)
        // If the mouth is open, move on to the next state
        if (message.is_mouth_open) {
          readyForBite()
        }
      }
    },
    [setDetectedMouthCenter, setMouthDetected, readyForBite]
  )
  useEffect(() => {
    let topic = subscribeToROSTopic(ros.current, FACE_DETECTION_TOPIC, FACE_DETECTION_TOPIC_MSG, faceDetectionCallback)
    /**
     * In practice, because the values passed in in the second argument of
     * useEffect will not change on re-renders, this return statement will
     * only be called when the component unmounts.
     */
    return () => {
      unsubscribeFromROSTopic(topic, faceDetectionCallback)
    }
  }, [faceDetectionCallback])

  /**
   * Create the ROS Service. This is created in local state to avoid re-creating
   * it upon every re-render.
   */
  let { serviceName, messageType } = ROS_SERVICE_NAMES[MEAL_STATE.U_BiteInitiation]
  let toggleFaceDetectionService = useRef(createROSService(ros.current, serviceName, messageType))

  /**
   * Toggles face detection on the first time this component is rendered, but
   * not upon additional re-renders. See here for more details on how `useEffect`
   * achieves this goal: https://stackoverflow.com/a/69264685
   */
  useEffect(() => {
    // Create a service request
    let request = createROSServiceRequest({ turn_on: true })
    // Call the service
    let service = toggleFaceDetectionService.current
    service.callService(request, (response) => console.log('Got service response', response))

    /**
     * In practice, because the values passed in in the second argument of
     * useEffect will not change on re-renders, this return statement will
     * only be called when the component unmounts.
     */
    return () => {
      // Create a service request
      let request = createROSServiceRequest({ turn_on: false })
      // Call the service
      service.callService(request, (response) => console.log('Got service response', response))
    }
  }, [toggleFaceDetectionService])

  /**
   * Callback function for when the user wants to move above plate.
   */
  const cancelBite = useCallback(() => {
    console.log('cancelBite')
    setMealState(MEAL_STATE.R_MovingAbovePlate)
  }, [setMealState])

  // Get the size of the robot's live video stream.
  const margin = convertRemToPixels(1)
  let { width, height } = scaleWidthHeightToWindow(REALSENSE_WIDTH, REALSENSE_HEIGHT, margin, margin, margin, margin)

  // Render the component
  return (
    <div style={{ display: 'block', width: '100%', height: '115vh', overflowX: 'hidden', overflowY: 'auto' }} className='outer'>
      {/* Tell the user whether their mouth has been detected or not */}
      {mouthDetected ? (
        <>
          <p className='transitionMessage' style={{ marginBottom: '0px' }}>
            Detected mouth.!
          </p>
          <p className='transitionMessage' style={{ marginBottom: '0px' }}>
            Open mouth when ready...
          </p>
        </>
      ) : (
        <p className='transitionMessage' style={{ marginBottom: '0px' }}>
          Detecting mouth...
        </p>
      )}

      {/* Show the user the face detection result */}
      <center>
        <img
          src={`${props.webVideoServerURL}/stream?topic=${FACE_DETECTION_IMG_TOPIC}&width=${Math.round(width)}&height=${Math.round(
            height
          )}&quality=20`}
          alt='Live video feed from the robot'
          style={{ width: width, height: height, display: 'block' }}
        />
      </center>

      {/*
       * If the mouth has been detected, let the user indicate whether they're
       * ready for the bite.
       */}
      {mouthDetected ? (
        <>
          <p className='transitionMessage' style={{ marginBottom: '0px' }}>
            ...or click this button.
          </p>
          <Row className='justify-content-center mx-auto my-2 w-75'>
            <Button
              variant='success'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              onClick={readyForBite}
              style={{ width: '300px', height: '200px' }}
            >
              <img src={moveToMouthImage} alt='move_to_mouth_image' className='center' />
            </Button>
          </Row>
          <Row className='justify-content-center mx-auto mt-5'>
            {/* Ask the user whether they want to move to above plate position */}
            <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: '140%' }}>
              Cancel bite and move above plate.
            </p>
            {/* Icon to move above plate */}
            <Button
              variant='warning'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              onClick={cancelBite}
              style={{ width: '300px', height: '200px' }}
            >
              <img src={moveAbovePlateImage} alt='move_above_plate_image' className='center' />
            </Button>
          </Row>
        </>
      ) : (
        <></>
      )}
      {/* If the user is running in debug mode, give them the option to skip */}
      {props.debug ? (
        <Button variant='secondary' className='justify-content-center mx-2 mb-2' size='lg' onClick={readyForBite}>
          Continue (Debug Mode)
        </Button>
      ) : (
        <></>
      )}
    </div>
  )
}
BiteInitiation.propTypes = {
  /**
   * Whether to run it in debug mode (e.g., if you aren't simulatenously running
   * the robot) or not
   */
  debug: PropTypes.bool.isRequired,
  // The URL of the web video server
  webVideoServerURL: PropTypes.string.isRequired
}

export default BiteInitiation
