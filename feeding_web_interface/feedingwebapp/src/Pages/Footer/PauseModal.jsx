// React imports
import React, { useRef } from 'react'
import Button from 'react-bootstrap/Button'
// The Modal is a screen that appears on top of the main app, and can be toggled
// on and off.
import Modal from 'react-bootstrap/Modal'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'

/**
 * The PauseModal gives the user the option to resume the robot when they are
 * ready.
 */
function PauseModal(props) {
  const ref = useRef(null)
  return (
    <>
      {/**
       * TODO:
       *   - Center the resume button in the modal so it works for all screen widths.
       *   - Determine why this Modal doesn't show an x-mark whereas LiveVideoModal
       *     does (probably has something to do with modal content being too big?)
       */}
      <Modal
        show={props.show}
        onHide={props.resumeClicked}
        size='lg'
        aria-labelledby='contained-modal-title-vcenter'
        backdrop='static'
        keyboard={false}
        centered
        id='pauseModal'
        fullscreen={true}
        ref={ref}
      >
        <Modal.Header>
          <Modal.Title id='contained-modal-title-vcenter'>⏸️ Paused!</Modal.Title>
        </Modal.Header>
        <Modal.Body style={{ paddingLeft: '10px', overflow: 'hidden' }}>
          <p className='transitionMessage' style={{ marginBottom: '10px', marginTop: '0px', fontSize: '24px' }}>
            Resume the feeding session when you are ready.
          </p>
          <Button
            className='bg-warning rounded btn-hugeE'
            style={{ fontSize: '50px', marginLeft: '8%' }}
            size='lg'
            onClick={props.resumeClicked}
          >
            ▶️ Resume
          </Button>
        </Modal.Body>
      </Modal>
    </>
  )
}
PauseModal.propTypes = {
  show: PropTypes.bool,
  resumeClicked: PropTypes.func
}

export default PauseModal
