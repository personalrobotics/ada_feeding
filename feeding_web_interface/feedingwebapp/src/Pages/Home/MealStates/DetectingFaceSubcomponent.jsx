// React Imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import PropTypes from 'prop-types'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'

// Local Imports
import { useROS, createROSService, createROSServiceRequest, subscribeToROSTopic, unsubscribeFromROSTopic } from '../../../ros/ros_helpers'
import '../Home.css'
import { convertRemToPixels } from '../../../helpers'
import { MEAL_STATE } from '../../GlobalState'
import { FACE_DETECTION_IMG_TOPIC, FACE_DETECTION_TOPIC, FACE_DETECTION_TOPIC_MSG, ROS_SERVICE_NAMES } from '../../Constants'
import VideoFeed from '../VideoFeed'

/**
 * The DetectingFace component appears after the robot has moved to the staging
 * configuration. It displays the output of face detection, and automatically
 * moves on to `R_MovingToMouth` when a face is detected.
 */
const DetectingFaceSubcomponent = (props) => {
  // Keep track of whether a mouth has been detected or not
  const [mouthDetected, setMouthDetected] = useState(false)
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Font size for text
  let textFontSize = 3
  let sizeSuffix = isPortrait ? 'vh' : 'vw'
  // The min and max distance from the camera to the face for the face to be
  // conidered valid. NOTE: This must match the values in the MoveToMouth tree.
  const min_face_distance = 0.4
  const max_face_distance = 1.25
  // Margin for the video feed and between the mask buttons. Note this cannot
  // be re-defined per render, otherwise it messes up re-rendering order upon
  // resize in VideoFeed.
  const margin = useMemo(() => convertRemToPixels(1), [])

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
      console.log('Got face detection message', message)
      let faceDetectedCallback = props.faceDetectedCallback
      if (message.is_face_detected) {
        let distance =
          (message.detected_mouth_center.point.x ** 2.0 +
            message.detected_mouth_center.point.y ** 2.0 +
            message.detected_mouth_center.point.z ** 2.0) **
          0.5
        if (distance > min_face_distance && distance < max_face_distance) {
          setMouthDetected(true)
          faceDetectedCallback()
        }
      }
    },
    [props.faceDetectedCallback, setMouthDetected]
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

  // Render the component
  return (
    <>
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
          {mouthDetected ? 'Mouth detected!' : 'Waiting to detect mouth...'}
        </p>
      </View>
      <View
        style={{
          flex: 9,
          alignItems: 'center',
          width: '100%',
          height: '100%'
        }}
      >
        <VideoFeed
          marginTop={margin}
          marginBottom={margin}
          marginLeft={margin}
          marginRight={margin}
          topic={FACE_DETECTION_IMG_TOPIC}
          webrtcURL={props.webrtcURL}
        />
      </View>
    </>
  )
}
DetectingFaceSubcomponent.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired,
  // The function to call, with the faceDetection message as an argument, when
  // a face is detected within the correct distance range.
  faceDetectedCallback: PropTypes.func.isRequired
}
export default DetectingFaceSubcomponent
