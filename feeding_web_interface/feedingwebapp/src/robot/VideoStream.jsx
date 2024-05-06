// React imports
import React, { useCallback, useEffect, useMemo, useRef } from 'react'
import PropTypes from 'prop-types'

// Local imports
import { useROS, subscribeToROSTopic, unsubscribeFromROSTopic } from '../ros/ros_helpers'
import { WebRTCConnection } from '../webrtc/webrtc_helpers'
import { REALSENSE_WIDTH, REALSENSE_HEIGHT } from '../Pages/Constants'

function dataURItoBlob(dataURI) {
  // Adapted from https://stackoverflow.com/a/38788279

  // convert base64 to raw binary data held in a string
  // doesn't handle URLEncoded DataURIs - see SO answer #6850276 for code that does this
  var splitString = dataURI.split(',')
  var byteString = atob(splitString[1])

  // separate out the mime component
  var mimeString = splitString[0].split(':')[1].split(';')[0]

  // write the bytes of the string to an ArrayBuffer
  var ab = new ArrayBuffer(byteString.length)
  var ia = new Uint8Array(ab)
  for (var i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i)
  }

  // write the ArrayBuffer to a blob, and you're done
  var blob = new Blob([ab], { type: mimeString })
  return blob
}

/**
 * Renders a video stream from the robot.
 *
 * NOTE: This page *must* be rendered on the robot, otherwise it will have
 * incredible lag.
 *
 * @param {string} webrtcURL - The URL of the webrtc signalling server.
 * @param {string} topic - The topic to subscribe to.
 *
 * @returns {object} A canvas element that displays the video stream.
 */
function VideoStream(props) {
  const canvas = useRef(null)
  const img = useMemo(() => document.createElement('img'), [])

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * Callback for when this page recieves an image from the robot.
   */
  const imageCallback = useCallback(
    (message) => {
      // console.log('Got image message for topic', props.topic)
      if (img.src) {
        URL.revokeObjectURL(img.src)
      }
      img.src = URL.createObjectURL(dataURItoBlob('data:image/jpg;base64,' + message.data))
    },
    [img /*, props.topic*/]
  )

  // Subscribe to the image topic
  useEffect(() => {
    console.log('subscribing to img topic', props.topic)
    let topic = subscribeToROSTopic(ros.current, props.topic, 'sensor_msgs/CompressedImage', imageCallback)
    const cleanup = () => {
      console.log('unsubscribing from img topic', props.topic)
      unsubscribeFromROSTopic(topic, imageCallback)
    }
    window.addEventListener('beforeunload', cleanup)
    /**
     * In practice, because the values passed in in the second argument of
     * useEffect will not change on re-renders, this return statement will
     * only be called when the component unmounts.
     */
    return () => {
      window.removeEventListener('beforeunload', cleanup)
      cleanup()
    }
  }, [imageCallback, props.topic, ros])

  // Draw the image on the canvas.
  const drawImage = useCallback(() => {
    if (img.src) {
      // Get the context
      const ctx = canvas.current.getContext('2d')
      // Draw the image
      ctx.drawImage(img, 0, 0, props.width, props.height)
    }
    requestAnimationFrame(drawImage)
  }, [canvas, img, props.width, props.height])

  // Create the WebRTC peer connection. Refresh it every refreshRateHz.
  useEffect(() => {
    // Get the canvas stream
    const stream = canvas.current.captureStream()

    // Create the peer connection
    const webRTCConnection = new WebRTCConnection({
      url: props.webrtcURL + '/publish',
      topic: props.topic,
      stream: stream
    })

    // Draw the image on the canvas
    drawImage()

    return () => {
      webRTCConnection.close()
    }
  }, [canvas, drawImage, props.topic, props.webrtcURL])

  return <canvas ref={canvas} width={props.width} height={props.height} style={{ position: 'absolute' }} />
}
VideoStream.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired,
  // The topic to subscribe to
  topic: PropTypes.string.isRequired,
  // The frame rate for the stream
  streamFPS: PropTypes.number.isRequired,
  // The desired width and height of the stream
  width: PropTypes.number.isRequired,
  height: PropTypes.number.isRequired
}
VideoStream.defaultProps = {
  streamFPS: 10,
  width: REALSENSE_WIDTH,
  height: REALSENSE_HEIGHT
}

export default VideoStream
