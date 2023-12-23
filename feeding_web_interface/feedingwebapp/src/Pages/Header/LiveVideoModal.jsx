// React imports
import React, { useMemo, useRef } from 'react'
import { useMediaQuery } from 'react-responsive'
// The Modal is a screen that appears on top of the main app, and can be toggled
// on and off.
import Modal from 'react-bootstrap/Modal'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'

// Local imports
import { convertRemToPixels } from '../../helpers'
import VideoFeed from '../Home/VideoFeed'

/**
 * The LiveVideoModal displays to the user the live video feed from the robot.
 *
 * TODO: Consider what will happen if the connection to ROS isn't working.
 */
function LiveVideoModal(props) {
  // Variables to render the VideoFeed
  const modalBodyRef = useRef(null)
  // Margin for the video feed and between the mask buttons. Note this cannot
  // be re-defined per render, otherwise it messes up re-rendering order upon
  // resize in VideoFeed.
  const margin = useMemo(() => convertRemToPixels(1), [])
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
      <Modal.Body ref={modalBodyRef} style={{ overflow: 'hidden' }}>
        <center>
          <VideoFeed
            topic='/local/camera/color/image_raw/compressed'
            updateRateHz={10}
            parent={modalBodyRef}
            marginTop={margin}
            marginBottom={margin}
            marginLeft={margin}
            marginRight={margin}
          />
        </center>
      </Modal.Body>
    </Modal>
  )
}
LiveVideoModal.propTypes = {
  // Whether or not the modal is visible
  show: PropTypes.bool.isRequired,
  // Callback function for when the modal is hidden
  onHide: PropTypes.func.isRequired
}

export default LiveVideoModal
