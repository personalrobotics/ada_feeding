// React imports
import React, { useCallback } from 'react'
import { MDBFooter } from 'mdb-react-ui-kit'
import Button from 'react-bootstrap/Button'
import { View } from 'react-native'
import Row from 'react-bootstrap/Row'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'

// Local imports
import { FOOTER_STATE_ICON_DICT } from '../Constants'
import { useGlobalState } from '../GlobalState'

/**
 * The Footer shows a pause button. When users click it, the app tells the robot
 * to immediately pause and displays a back button that allows them to return to
 * previous state and a resume button that allows them to resume current state.
 */
const Footer = (props) => {
  // Get the current meal state
  const mealState = useGlobalState((state) => state.mealState)

  // Icons and other parameters for the footer buttons
  let pauseIcon = '/robot_state_imgs/pause_button_icon.svg'
  let backIcon = props.backMealState ? FOOTER_STATE_ICON_DICT[props.backMealState] : ''
  let resumeIcon = FOOTER_STATE_ICON_DICT[mealState]
  let phantomButtonIcon = '/robot_state_imgs/phantom_view_image.svg'
  // Width of Back and Resume buttons
  let backResumeButtonWidth = '150px'
  // Height of all Footer buttons
  let footerButtonHight = '100px'
  /**
   * When the pause button is clicked, execute the callback and display the
   * back and resume buttons.
   */
  const pauseClicked = useCallback(() => {
    props.pauseCallback()
    props.setPaused(true)
  }, [])

  /**
   * When the resume button is clicked, execute the callback and display the
   * pause button.
   */
  const resumeClicked = useCallback(() => {
    if (props.resumeCallback) {
      props.resumeCallback()
    }
    props.setPaused(false)
  }, [])

  /**
   * When the back button is clicked, execute the callback and display the pause
   * button.
   */
  const backClicked = useCallback(() => {
    if (props.backCallback) {
      props.backCallback()
    }
    props.setPaused(false)
  }, [])

  /**
   * Get the pause text and button to render in footer.
   *
   * @returns {JSX.Element} the pause text and button
   */
  const renderPauseButton = useCallback(() => {
    return (
      <>
        <Row className='justify-content-center mx-auto'>
          <p className='transitionMessage' style={{ marginBottom: '0', fontSize: '170%', color: 'white', fontWeight: 'bold' }}>
            ⏸️ Pause
          </p>
          {/* Icon to pause */}
          <Button
            variant='danger'
            onClick={pauseClicked}
            style={{ marginLeft: '10', marginRight: '10', marginTop: '0', width: '350px', height: { footerButtonHight } }}
          >
            <img style={{ width: '135px', height: '90px' }} src={pauseIcon} alt='pause_icon' className='center' />
          </Button>
        </Row>
      </>
    )
  }, [pauseClicked])

  /**
   * Get the back text and button to render in footer.
   *
   * @returns {JSX.Element} the back text and button
   */
  const renderBackButton = useCallback(() => {
    return (
      <>
        <p className='transitionMessage' style={{ marginBottom: '0', fontSize: '170%', color: 'white', fontWeight: 'bold' }}>
          ◀️ Back
        </p>
        {/* Icon to move to previous state */}
        <Button
          variant='warning'
          onClick={backClicked}
          style={{ marginLeft: 10, marginRight: 10, width: { backResumeButtonWidth }, height: { footerButtonHight } }}
        >
          <img style={{ width: '120px', height: '72px' }} src={backIcon} alt='back_icon' className='center' />
        </Button>
      </>
    )
  }, [backClicked])

  /**
   * Get the resume text and button to render in footer.
   *
   * @returns {JSX.Element} the resume text and button
   */
  const renderResumeButton = useCallback(() => {
    return (
      <>
        <p
          className='transitionMessage'
          style={{ marginBottom: '0', fontSize: '170%', color: 'white', fontWeight: 'bold', textAlign: 'right' }}
        >
          ▶️ Resume
        </p>
        {/* Icon to resume current state */}
        <Button
          variant='success'
          onClick={resumeClicked}
          style={{ marginLeft: 10, marginRight: 10, width: { backResumeButtonWidth }, height: { footerButtonHight } }}
        >
          <img style={{ width: '120px', height: '72px' }} src={resumeIcon} alt='resume_icon' className='center' />
        </Button>
      </>
    )
  }, [resumeClicked])

  /**
   * Get the phantom view to render in footer. This is used as a placeholder
   * when the back button or resume button are disabled.
   *
   * @returns {JSX.Element} the phantom view
   */
  let renderPhantomButton = function () {
    return (
      <>
        <Button
          variant='ghost'
          disabled={true}
          style={{
            backgroundColor: 'transparent',
            border: 'none',
            marginLeft: 10,
            marginRight: 10,
            width: { backResumeButtonWidth },
            height: { footerButtonHight }
          }}
        >
          <img style={{ width: '120px', height: '72px' }} src={phantomButtonIcon} alt='phantom_button_img' className='center' />
        </Button>
      </>
    )
  }

  // Render the component
  return (
    <>
      <MDBFooter bgColor='dark' className='text-center text-lg-left fixed-bottom'>
        <div className='text-center p-3' style={{ backgroundColor: 'rgba(0, 0, 0, 0.2)' }}>
          {props.paused ? (
            <View style={{ flexDirection: 'row', justifyContent: 'center' }}>
              <View>{props.backCallback && props.backMealState ? renderBackButton() : renderPhantomButton()}</View>
              <View>{props.resumeCallback ? renderResumeButton() : renderPhantomButton()}</View>
            </View>
          ) : (
            renderPauseButton()
          )}
        </div>
      </MDBFooter>
    </>
  )
}
Footer.propTypes = {
  paused: PropTypes.bool.isRequired,
  setPaused: PropTypes.func.isRequired,
  pauseCallback: PropTypes.func.isRequired,
  // If any of the below three are null, the Footer won't render that button
  resumeCallback: PropTypes.func,
  backCallback: PropTypes.func,
  backMealState: PropTypes.string
}

export default Footer
