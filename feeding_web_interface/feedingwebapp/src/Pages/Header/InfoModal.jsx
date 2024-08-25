// React imports
import React, { useCallback, useMemo, useState } from 'react'
import { useMediaQuery } from 'react-responsive'
import Modal from 'react-bootstrap/Modal'
import PropTypes from 'prop-types'
import { ToastContainer, toast } from 'react-toastify'
import { View } from 'react-native'

// Local imports
import { CAMERA_FEED_TOPIC, MODAL_CONTAINER_ID } from '../Constants'
import { useGlobalState } from '../GlobalState'
import TeleopSubcomponent from './TeleopSubcomponent'
import VideoFeed from '../Home/VideoFeed'
import ToggleButtonGroup from '../../buttons/ToggleButtonGroup'

/**
 * The InfoModal displays to the user the live video feed from the robot.
 */
function InfoModal(props) {
  // The three different modes of the info modal
  const VIDEO_MODE = useMemo(() => 'Video', [])
  const TELEOP_MODE = useMemo(() => 'Teleop', [])
  const SYSTEM_STATUS_MODE = useMemo(() => 'Status', [])
  const [mode, setMode] = useState(VIDEO_MODE)

  // Teleop don't allow changing to teleop mode if app is in a moving state
  const inNonMovingState = useGlobalState((state) => state.inNonMovingState)
  const teleopCallback = useCallback(() => {
    /**
     * TODO: We need a more dynamic way to determing if the app is in a non-moving
     * state, i.e., if robot motion is in error or if auto-continue is not going
     * to get triggered or such.
     */
    if (inNonMovingState) {
      setMode(TELEOP_MODE)
    } else {
      toast.info('Cannot switch to teleop until the app is on a non-moving page.', {
        containerId: MODAL_CONTAINER_ID,
        toastId: 'noTeleop'
      })
    }
  }, [inNonMovingState, TELEOP_MODE])

  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  const flexDirection = isPortrait ? 'column' : 'row'
  const otherDirection = isPortrait ? 'row' : 'column'
  // Text font size for portrait and landscape orientations
  let textFontSize = isPortrait ? '3vh' : '6vh'

  /**
   * When the InfoModal is closed, switch to Video mode and call the parent-specified
   * callback.
   */
  const onModalHide = useCallback(() => {
    setMode(VIDEO_MODE)
    let onHide = props.onHide
    onHide()
  }, [props.onHide, setMode, VIDEO_MODE])

  return (
    <Modal
      show={props.show}
      onHide={onModalHide}
      size='lg'
      aria-labelledby='contained-modal-title-vcenter'
      backdrop='static'
      keyboard={false}
      centered
      id='InfoModal'
      fullscreen={true}
    >
      <Modal.Header closeButton>
        <Modal.Title id='contained-modal-title-vcenter' style={{ fontSize: textFontSize }}>
          Robot Information
        </Modal.Title>
      </Modal.Header>
      <Modal.Body
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: flexDirection,
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%',
          overflow: 'hidden'
        }}
      >
        {/**
         * The ToastContainer is an alert that pops up on the top of the screen
         * and has a timeout.
         */}
        <ToastContainer style={{ fontSize: textFontSize }} containerId={MODAL_CONTAINER_ID} enableMultiContainer={true} />
        <View
          style={{
            flex: 3,
            flexDirection: otherDirection,
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          {/* Switch between the different modes. */}
          <ToggleButtonGroup
            valueOptions={[VIDEO_MODE, TELEOP_MODE, SYSTEM_STATUS_MODE]}
            currentValue={mode}
            callbackFunctions={[setMode, teleopCallback, setMode]}
            horizontal={isPortrait}
          />
        </View>
        <View
          style={{
            flex: 18,
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          {/* Display the main contents of the page */}
          {mode === VIDEO_MODE ? (
            <VideoFeed topic={CAMERA_FEED_TOPIC} updateRateHz={10} webrtcURL={props.webrtcURL} />
          ) : mode === TELEOP_MODE ? (
            <TeleopSubcomponent allowIncreasingForceThreshold={true} allowRetaringFTSensor={true} />
          ) : mode === SYSTEM_STATUS_MODE ? (
            <div>System Status</div>
          ) : (
            <></>
          )}
        </View>
      </Modal.Body>
    </Modal>
  )
}
InfoModal.propTypes = {
  // Whether or not the modal is visible
  show: PropTypes.bool.isRequired,
  // Callback function for when the modal is hidden
  onHide: PropTypes.func.isRequired,
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default InfoModal
