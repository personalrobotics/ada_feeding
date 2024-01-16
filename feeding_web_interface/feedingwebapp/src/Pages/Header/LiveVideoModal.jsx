// React imports
import React from 'react'
import { useMediaQuery } from 'react-responsive'
// The Modal is a screen that appears on top of the main app, and can be toggled
// on and off.
import Modal from 'react-bootstrap/Modal'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'

// Local imports
import { CAMERA_FEED_TOPIC } from '../Constants'
import VideoFeed from '../Home/VideoFeed'

/**
 * The LiveVideoModal displays to the user the live video feed from the robot.
 *
 * TODO: Consider what will happen if the connection to ROS isn't working.
 */
function LiveVideoModal(props) {
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Text font size for portrait and landscape orientations
  let textFontSize = isPortrait ? '3vh' : '6vh'

  return (
    <Modal
      show={props.show}
      onHide={props.onHide}
      size='lg'
      aria-labelledby='contained-modal-title-vcenter'
      backdrop='static'
      keyboard={false}
      centered
      id='liveVideoModal'
      fullscreen={true}
    >
      <Modal.Header closeButton>
        <Modal.Title id='contained-modal-title-vcenter' style={{ fontSize: textFontSize }}>
          Live Video
        </Modal.Title>
      </Modal.Header>
      <Modal.Body
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%',
          overflow: 'hidden'
        }}
      >
        <VideoFeed topic={CAMERA_FEED_TOPIC} updateRateHz={10} webrtcURL={props.webrtcURL} />
      </Modal.Body>
    </Modal>
  )
}
LiveVideoModal.propTypes = {
  // Whether or not the modal is visible
  show: PropTypes.bool.isRequired,
  // Callback function for when the modal is hidden
  onHide: PropTypes.func.isRequired,
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default LiveVideoModal
