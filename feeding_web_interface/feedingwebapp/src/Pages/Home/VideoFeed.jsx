// React Imports
import React, { useCallback, useEffect, useRef, useState } from 'react'
import Button from 'react-bootstrap/Button'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local Imports
import { CAMERA_FEED_TOPIC, REALSENSE_WIDTH, REALSENSE_HEIGHT } from '../Constants'
import { useWindowSize } from '../../helpers'
import { WebRTCConnection } from '../../webrtc/webrtc_helpers'

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
  let textFontSize = '2.5vh'

  /**
   * Create the peer connection
   */
  useEffect(() => {
    // Create the peer connection
    console.log('Creating peer connection', props.webrtcURL, refreshCount)
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
      webRTCConnection.close()
    }
  }, [props.topic, props.webrtcURL, refreshCount, videoRef])

  // Callback to resize the image based on the parent width and height
  const resizeImage = useCallback(() => {
    console.log('Resizing image', parentRef.current)
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
    setImgWidth(childWidth)
    setImgHeight(childHeight)
    setScaleFactor(childScaleFactor)
  }, [parentRef, props.marginTop, props.marginBottom, props.marginLeft, props.marginRight])

  /** When the resize event is triggered, the elements have not yet been laid out,
   * and hence the parent width/height might not be accurate yet based on the
   * specified flex layout. Hence, we wait until the next event cycle to resize
   * the video feed.
   */
  const resizeImageNextEventCycle = useCallback(() => {
    setTimeout(resizeImage, 0)
  }, [resizeImage])
  useWindowSize(resizeImageNextEventCycle)

  // When the component is first mounted, resize the image
  useEffect(() => {
    resizeImageNextEventCycle()
  }, [resizeImageNextEventCycle])

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

  // Render the component
  return (
    <>
      <View
        ref={parentRef}
        style={{
          flex: 4,
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%'
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
          // TODO: consider replacing 'center' with 'end' if the margin is 0.
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
            fontSize: textFontSize,
            width: '60%',
            color: 'black'
          }}
          onClick={() => setRefreshCount(refreshCount + 1)}
        >
          Reload Video
        </Button>
      </View>
    </>
  )
}
VideoFeed.propTypes = {
  // The margins around the video feed
  marginTop: PropTypes.number,
  marginBottom: PropTypes.number,
  marginLeft: PropTypes.number,
  marginRight: PropTypes.number,
  // The topic of the video feed
  topic: PropTypes.string.isRequired,
  /**
   * An optional callback function for when the user clicks on the video feed.
   * This function should take in two parameters, `x` and `y`, which are the
   * coordinates of the click in the **unscaled** image (e.g., the image of
   * size REALSENSE_WIDTH x REALSENSE_HEIGHT).
   */
  pointClicked: PropTypes.func,
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}
VideoFeed.defaultProps = {
  marginTop: 0,
  marginBottom: 0,
  marginLeft: 0,
  marginRight: 0,
  topic: CAMERA_FEED_TOPIC
}

export default VideoFeed
