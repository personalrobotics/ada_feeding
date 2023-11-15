// React Imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Button from 'react-bootstrap/Button'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'

// Local Imports
import { useROS, createROSService, createROSServiceRequest, subscribeToROSTopic, unsubscribeFromROSTopic } from '../../../ros/ros_helpers'
import '../Home.css'
import { convertRemToPixels } from '../../../helpers'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import {
  FACE_DETECTION_IMG_TOPIC,
  FACE_DETECTION_TOPIC,
  FACE_DETECTION_TOPIC_MSG,
  MOVING_STATE_ICON_DICT,
  ROS_SERVICE_NAMES
} from '../../Constants'
import VideoFeed from '../VideoFeed'

/**
 * The DetectingFace component appears after the robot has moved to the staging
 * configuration. It displays the output of face detection, and automatically
 * moves on to `R_MovingToMouth` when a face is detected.
 */
const DetectingFace = (props) => {
  // Keep track of whether a mouth has been detected or not
  const [mouthDetected, setMouthDetected] = useState(false)
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  const setMoveToMouthActionGoal = useGlobalState((state) => state.setMoveToMouthActionGoal)
  const faceDetectionAutoContinue = useGlobalState((state) => state.faceDetectionAutoContinue)
  const setFaceDetectionAutoContinue = useGlobalState((state) => state.setFaceDetectionAutoContinue)
  // Get icon image for move to mouth
  let moveToMouthImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth]
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
  // The min and max distance from the camera to the face for the face to be
  // conidered valid. NOTE: This must match the values in the MoveToMouth tree.
  const min_face_distance = 0.4
  const max_face_distance = 1.25
  // Margin for the video feed and between the mask buttons. Note this cannot
  // be re-defined per render, otherwise it messes up re-rendering order upon
  // resize in VideoFeed.
  const margin = useMemo(() => convertRemToPixels(1), [])
  // Reference to the DOM element of the parent of the video feed
  const videoParentRef = useRef(null)

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

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
   * Subscribe to the ROS Topic with the face detection result. This is created
   * in local state to avoid re-creating it upon every re-render.
   */
  const faceDetectionCallback = useCallback(
    (message) => {
      console.log('Got face detection message', message)
      if (message.is_face_detected) {
        let distance =
          (message.detected_mouth_center.point.x ** 2.0 +
            message.detected_mouth_center.point.y ** 2.0 +
            message.detected_mouth_center.point.z ** 2.0) **
          0.5
        if (distance > min_face_distance && distance < max_face_distance) {
          setMouthDetected(true)
          setMoveToMouthActionGoal({
            face_detection: message
          })
          // Automatically move on to the next stage if a face is detected
          if (faceDetectionAutoContinue) {
            moveToMouthCallback()
          }
        }
      }
    },
    [faceDetectionAutoContinue, moveToMouthCallback, setMoveToMouthActionGoal]
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
  let { serviceName, messageType } = ROS_SERVICE_NAMES[MEAL_STATE.R_DetectingFace]
  let toggleFaceDetectionService = useRef(createROSService(ros.current, serviceName, messageType))

  /**
   * Toggles face detection on the first time this component is rendered, but
   * not upon additional re-renders. See here for more details on how `useEffect`
   * achieves this goal: https://stackoverflow.com/a/69264685
   */
  useEffect(() => {
    // Create a service request
    let request = createROSServiceRequest({ data: true })
    // Call the service
    let service = toggleFaceDetectionService.current
    service.callService(request, (response) => console.log('Got toggle face detection service response', response))

    /**
     * In practice, because the values passed in in the second argument of
     * useEffect will not change on re-renders, this return statement will
     * only be called when the component unmounts.
     */
    return () => {
      // Create a service request
      let request = createROSServiceRequest({ data: false })
      // Call the service
      service.callService(request, (response) => console.log('Got toggle face detection service response', response))
    }
  }, [toggleFaceDetectionService])

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
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
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
                {mouthDetected ? 'Mouth detected!' : 'Waiting to detect mouth...'}
              </p>
            </View>
            <View
              ref={videoParentRef}
              style={{
                flex: 9,
                alignItems: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              <VideoFeed
                webVideoServerURL={props.webVideoServerURL}
                parent={videoParentRef}
                marginTop={margin}
                marginBottom={margin}
                marginLeft={margin}
                marginRight={margin}
                topic={FACE_DETECTION_IMG_TOPIC}
                type='mjpeg&quality=20'
              />
            </View>
          </View>
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
            <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
              {mouthDetected ? 'Continue' : 'Continue without detecting mouth'}
            </p>
            {/* Icon to move to mouth position */}
            <Button
              variant='success'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              onClick={moveToMouthCallback}
              style={{ width: buttonWidth, height: buttonHeight }}
            >
              <img src={moveToMouthImage} alt='move_to_mouth_image' className='center' style={{ width: iconWidth, height: iconHeight }} />
            </Button>
          </View>
        </View>
      </>
    )
  }, [
    dimension,
    margin,
    mouthDetected,
    moveToMouthCallback,
    moveToMouthImage,
    props.webVideoServerURL,
    textFontSize,
    buttonHeight,
    buttonWidth,
    iconHeight,
    iconWidth,
    faceDetectionAutoContinue,
    setFaceDetectionAutoContinue
  ])

  // Render the component
  return fullPageView()
}
DetectingFace.propTypes = {
  /**
   * Whether to run it in debug mode (e.g., if you aren't simulatenously running
   * the robot) or not
   */
  debug: PropTypes.bool.isRequired,
  // The URL of the web video server
  webVideoServerURL: PropTypes.string.isRequired
}

export default DetectingFace
