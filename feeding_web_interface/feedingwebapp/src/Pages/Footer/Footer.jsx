// React imports
import React, { useState } from 'react'
import { MDBFooter } from 'mdb-react-ui-kit'
import Button from 'react-bootstrap/Button'

// Local imports
import PauseModal from './PauseModal'

/**
 * The Footer shows a pause button. When users click it, the app tells the robot
 * to immediately pause and displays a model that gives them the option to resume.
 */
const Footer = () => {
  // Create a local state variable for whether the robot is paused
  const [pause, setPause] = useState(false)

  /**
   * When the resume button is clicked, close the modal and resume robot motion.
   */
  function resumeClicked() {
    setPause(false)
  }

  return (
    <>
      {/**
       * The footer has a big pause button, which shows the PauseModal when clicked.
       */}
      <MDBFooter bgColor='dark' className='text-center text-lg-left fixed-bottom'>
        <div className='text-center p-3' style={{ backgroundColor: 'rgba(0, 0, 0, 0.2)' }}>
          {
            <Button className='bg-warning rounded btn-hugeE' style={{ fontSize: '50px' }} size='lg' onClick={() => setPause(true)}>
              ⏸️ Pause
            </Button>
          }
        </div>
      </MDBFooter>

      {/**
       * The PauseModal toggles on and off with the Pause button and shows a
       * screen where the user can resume the robot.
       */}
      <PauseModal show={pause} resumeClicked={resumeClicked} />
    </>
  )
}

export default Footer
