/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React Imports
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useMediaQuery } from 'react-responsive'
import Button from 'react-bootstrap/Button'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local Imports
import { CAMERA_FEED_TOPIC, REALSENSE_WIDTH, REALSENSE_HEIGHT, ROS_SERVICE_NAMES } from '../Constants'
import { useWindowSize } from '../../helpers'
import { WebRTCConnection } from '../../webrtc/webrtc_helpers'
import { createROSService, createROSServiceRequest, useROS } from '../../ros/ros_helpers'
import { MEAL_STATE } from '../GlobalState'

/**
 * Takes in an imageWidth and imageHeight, and returns a width and height that
 * maintains the same aspect ratio but fits within the window.
 *
 * NOTE: if run in iOS on low power mode, the video element requires the user to press
 * "play" to start it, which is treated as a click.
 *
 * @param {number} parentWidth the width of the parent DOM element in pixels
 * @param {number} parentHeight the height of the parent DOM element in pixels
 * @param {number} imageWidth the original image's width in pixels
 * @param {number} imageHeight the original image's height in pixels
 * @param {number} marginTop the desired top margin between window and image, in pixels
 * @param {number} marginBottom the desired bottom margin between window and image, in pixels
 * @param {number} marginLeft the desired left margin between window and image, in pixels
 * @param {number} marginRight the desired right margin between window and image, in pixels
 *
 * @returns {object} the width and height of the image that fits within the window and has the requested margins
 */
function scaleWidthHeightToWindow(
  parentWidth,
  parentHeight,
  imageWidth,
  imageHeight,
  marginTop = 0,
  marginBottom = 0,
  marginLeft = 0,
  marginRight = 0
) {
  // Calculate the aspect ratio of the image
  let imageAspectRatio = imageWidth / imageHeight
  // Get the aspect ratio of the available subset of the window
  let availableWidth = parentWidth - marginLeft - marginRight
  let availableHeight = parentHeight - marginTop - marginBottom
  let availableAspectRatio = availableWidth / availableHeight

  // Calculate the width and height of the image that fits within the window
  let returnWidth, returnHeight
  if (availableAspectRatio > imageAspectRatio) {
    returnHeight = Math.round(availableHeight)
    returnWidth = Math.round(imageAspectRatio * returnHeight)
  } else {
    returnWidth = Math.round(availableWidth)
    returnHeight = Math.round(returnWidth / imageAspectRatio)
  }

  // Calculate the scale factor
  let scaleFactorWidth = returnWidth / imageWidth
  let scaleFactorHeight = returnHeight / imageHeight
  let scaleFactor = (scaleFactorWidth + scaleFactorHeight) / 2

  return {
    width: returnWidth,
    height: returnHeight,
    scaleFactor: scaleFactor
  }
}

/**
 * The VideoFeed component takes in a reference to the parent DOM element, and
 * displays a video feed that takes up the maximum size within the parent DOM,
 * while maintaining the aspect ratio of the video feed and respecting specified
 * margins.
 *
 * Note that for this to work, the parent DOM element must have a specified
 * height and width that does not change with its contents. One way to achieve
 * this is to ensure the `width` and `height` props of the parent are set.
 */
const VideoFeed = (props) => {
  // Local state variables to keep track of the width and height of the video feed
  const [imgWidth, setImgWidth] = useState(0)
  const [imgHeight, setImgHeight] = useState(0)
  const [scaleFactor, setScaleFactor] = useState(0.0)
  const [refreshCount, setRefreshCount] = useState(0)

  // Ref for the video element
  const videoRef = useRef(null)
  const parentRef = useRef(null)

  // Rendering variables
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  let textFontSize = isPortrait ? 2.5 : 3.0
  let sizeSuffix = 'vh'

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * Create the ROS Service Clients to toggle face detection.
   */
  let { serviceName, messageType } = ROS_SERVICE_NAMES[MEAL_STATE.R_DetectingFace]
  let toggleFaceDetectionService = useRef(createROSService(ros.current, serviceName, messageType))

  /**
   * Create the peer connection
   */
  useEffect(() => {
    // Toggle on face detection if specified
    let service = toggleFaceDetectionService.current
    if (props.toggleFaceDetection) {
      let request = createROSServiceRequest({ data: true })
      service.callService(request, (response) => console.log('VideoFeed got toggle face detection on service response', response))
    }

    // Create the peer connection
    console.log('Creating peer connection', props.webrtcURL, refreshCount, props.externalRefreshCount)
    const webRTCConnection = new WebRTCConnection({
      url: props.webrtcURL + '/subscribe',
      topic: props.topic,
      onTrackAdded: (event) => {
        console.log('Got track event', event)
        if (event.streams && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0]
          console.log('video', videoRef.current)
        }
      },
      transceiverKind: 'video',
      transceiverOptions: { direction: 'recvonly' }
    })

    return () => {
      // Close the peer connection
      webRTCConnection.close()

      // Toggle off face detection if specified
      if (props.toggleFaceDetection) {
        let request = createROSServiceRequest({ data: false })
        service.callService(request, (response) => console.log('VideoFeed got toggle face detection off service response', response))
      }
    }
  }, [
    props.externalRefreshCount,
    props.toggleFaceDetection,
    props.topic,
    props.webrtcURL,
    refreshCount,
    toggleFaceDetectionService,
    videoRef
  ])

  // Callback to resize the image based on the parent width and height
  const resizeImage = useCallback(
    (delay_ms = 10) => {
      if (!parentRef.current) {
        return
      }
      // Get the width and height of the parent DOM element
      let parentWidth = parentRef.current.clientWidth
      let parentHeight = parentRef.current.clientHeight

      // Calculate the width and height of the video feed
      let {
        width: childWidth,
        height: childHeight,
        scaleFactor: childScaleFactor
      } = scaleWidthHeightToWindow(
        parentWidth,
        parentHeight,
        REALSENSE_WIDTH,
        REALSENSE_HEIGHT,
        props.marginTop,
        props.marginBottom,
        props.marginLeft,
        props.marginRight
      )

      // Set the width and height of the video feed
      setImgWidth(childWidth * props.zoom)
      setImgHeight(childHeight * props.zoom)
      setScaleFactor(childScaleFactor * props.zoom)

      // If the width or height is zero, schedule another resize event in the next
      // event cycle. This is because initially the elements have not been laid out,
      // and it might take a few event cycles to do so.
      if (childWidth === 0.0 || childHeight === 0.0) {
        setTimeout(resizeImage, delay_ms)
      }
    },
    [parentRef, props.marginTop, props.marginBottom, props.marginLeft, props.marginRight, props.zoom]
  )

  // Resize the element when the window is resized
  useWindowSize(resizeImage)

  // When the component is first mounted and when the reload button is clicked,
  // resize the image
  useEffect(() => {
    console.log('Resizing image', refreshCount, props.externalRefreshCount)
    resizeImage()
  }, [props.externalRefreshCount, refreshCount, resizeImage])

  // The callback for when the image is clicked.
  const imageClicked = useCallback(
    (event) => {
      // Get the position of the click relative to the raw RealSense image.
      let rect = event.target.getBoundingClientRect()
      let x = event.clientX - rect.left // x position within the image.
      let y = event.clientY - rect.top // y position within the image.
      let x_raw = Math.round(x / scaleFactor) // x position within the raw image.
      let y_raw = Math.round(y / scaleFactor) // y position within the raw image.
      console.log('Button click on unscaled image: (' + x_raw + ', ' + y_raw + ')')
      let pointClicked = props.pointClicked

      // Call the callback function if it exists
      if (pointClicked) {
        pointClicked(x_raw, y_raw)
      }
    },
    [props.pointClicked, scaleFactor]
  )

  const renderZoomControls = useCallback(
    (zoom, setZoom, zoomMin, zoomMax) => {
      // Return three views, containing a button, -, to reduce the zoom, a button,
      // +, to increase the zoom, and text in-between that indicates the zoom
      // level.
      return (
        <>
          <View
            style={{
              flex: 1,
              alignItems: 'center',
              justifyContent: 'start',
              width: '100%',
              height: '100%'
            }}
          >
            <Button
              variant='warning'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              style={{
                fontSize: textFontSize.toString() + sizeSuffix,
                color: 'black'
              }}
              onClick={() => setZoom(Math.max(zoomMin, zoom - 0.1))}
              disabled={zoom <= zoomMin}
            >
              -
            </Button>
          </View>
          <View
            style={{
              flex: 2,
              alignItems: 'center',
              justifyContent: 'start',
              width: '60%',
              height: '100%'
            }}
          >
            <p
              style={{
                fontSize: textFontSize.toString() + sizeSuffix,
                color: 'black'
              }}
            >
              {Math.round(zoom * 100)}%
            </p>
          </View>
          <View
            style={{
              flex: 1,
              alignItems: 'center',
              justifyContent: 'start',
              width: '100%',
              height: '100%'
            }}
          >
            <Button
              variant='warning'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              style={{
                fontSize: textFontSize.toString() + sizeSuffix,
                color: 'black'
              }}
              onClick={() => setZoom(Math.min(zoomMax, zoom + 0.1))}
              disabled={zoom >= zoomMax}
            >
              +
            </Button>
          </View>
        </>
      )
    },
    [textFontSize, sizeSuffix]
  )

  // Render the component
  return (
    <View
      style={{
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'start',
        width: '100%',
        height: '100%'
      }}
    >
      <View
        ref={parentRef}
        style={{
          flex: 4,
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%',
          overflow: 'hidden'
        }}
      >
        <video
          playsInline
          autoPlay
          muted
          ref={videoRef}
          alt='Live video feed from the robot'
          style={{
            width: imgWidth,
            height: imgHeight,
            display: 'block',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          onClick={props.pointClicked ? imageClicked : null}
        />
      </View>
      <View
        style={{
          flex: 1,
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'start',
          width: '100%',
          height: '100%'
        }}
      >
        {props.setZoom ? renderZoomControls(props.zoom, props.setZoom, props.zoomMin, props.zoomMax) : <></>}
        <View
          style={{
            flex: 4,
            alignItems: 'center',
            justifyContent: 'start',
            width: '100%',
            height: '100%'
          }}
        >
          <Button
            variant='warning'
            className='mx-2 mb-2 btn-huge'
            size='lg'
            style={{
              fontSize: textFontSize.toString() + sizeSuffix,
              color: 'black'
            }}
            onClick={() => setRefreshCount((x) => x + 1)}
          >
            Reload Video
          </Button>
        </View>
      </View>
    </View>
  )
}
VideoFeed.propTypes = {
  // The margins around the video feed
  marginTop: PropTypes.number,
  marginBottom: PropTypes.number,
  marginLeft: PropTypes.number,
  marginRight: PropTypes.number,
  // A number that changes when some external entity wants this component to refresh.
  externalRefreshCount: PropTypes.number,
  // The topic of the video feed
  topic: PropTypes.string.isRequired,
  // Whether this component should toggle face detection on when it is mounted and
  // the reload button is clicked, and toggle it off when it is unmounted
  toggleFaceDetection: PropTypes.bool,
  /**
   * An optional callback function for when the user clicks on the video feed.
   * This function should take in two parameters, `x` and `y`, which are the
   * coordinates of the click in the **unscaled** image (e.g., the image of
   * size REALSENSE_WIDTH x REALSENSE_HEIGHT).
   */
  pointClicked: PropTypes.func,
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired,
  // How much to zoom the camera feed by
  zoom: PropTypes.number.isRequired,
  // Min/max zoom
  zoomMin: PropTypes.number.isRequired,
  zoomMax: PropTypes.number.isRequired,
  // If this is set, then VideoFeed renders buttons to allow users to change the zoom
  setZoom: PropTypes.func
}
VideoFeed.defaultProps = {
  marginTop: 0,
  marginBottom: 0,
  marginLeft: 0,
  marginRight: 0,
  externalRefreshCount: 0,
  topic: CAMERA_FEED_TOPIC,
  toggleFaceDetection: false,
  zoom: 1.0,
  zoomMin: 1.0,
  zoomMax: 2.0
}

export default VideoFeed
