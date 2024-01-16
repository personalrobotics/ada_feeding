// React imports
import React, { useCallback } from 'react'
import { MDBFooter } from 'mdb-react-ui-kit'
import Button from 'react-bootstrap/Button'
import { View } from 'react-native'
import Row from 'react-bootstrap/Row'
import { useMediaQuery } from 'react-responsive'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'
// Local imports
import { MOVING_STATE_ICON_DICT } from '../Constants'

/**
 * The Footer shows a pause button. When users click it, the app tells the robot
 * to immediately pause and displays a back button that allows them to return to
 * previous state and a resume button that allows them to resume current state.
 *
 * @param {string} mealState - the current meal state
 * @param {bool} paused - whether the robot is currently paused
 * @param {function} pauseCallback - callback function for when the pause button
 *     is clicked
 * @param {function} backCallback - callback function for when the back button
 *     is clicked. If null, don't render the back button.
 * @param {string} backMealState - the meal state to return to when the back
 *     button is clicked. If null, don't render the back button.
 * @param {function} resumeCallback - callback function for when the resume
 *     button is clicked. If null, don't render the resume button.
 */
const Footer = (props) => {
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Icons for the footer buttons
  let pauseIcon = '/robot_state_imgs/pause_button_icon.svg'
  let backIcon = props.backMealState ? MOVING_STATE_ICON_DICT[props.backMealState] : ''
  let resumeIcon = MOVING_STATE_ICON_DICT[props.mealState]
  // Sizes (width, height, fontsize) of footer buttons
  let pauseButtonWidth = '98vw'
  let backResumeButtonWidth = '47vw'
  let pauseFontSize = '7vh'
  let backResumeFontSize = isPortrait ? '3vh' : '7vh'
  let footerButtonHeight = '12vh'
  // Margins around footer buttons
  let footerLeftRightMargin = '1.6vh'
  let footerTopBottomMargin = '0.3vh'
  // A single nested object with all footer buttons' properties
  const buttonConfig = {
    pause: {
      text: 'Pause',
      icon: pauseIcon,
      disabled: false,
      variant: 'danger',
      callback: props.pauseCallback,
      buttonWidth: pauseButtonWidth,
      buttonHeight: footerButtonHeight,
      fontSize: pauseFontSize,
      iconSize: footerButtonHeight,
      backgroundColor: null
    },
    back: {
      text: 'Back',
      icon: backIcon,
      disabled: false,
      variant: 'warning',
      callback: props.backCallback,
      buttonWidth: backResumeButtonWidth,
      buttonHeight: footerButtonHeight,
      fontSize: backResumeFontSize,
      iconSize: footerButtonHeight,
      backgroundColor: null
    },
    resume: {
      text: 'Resume',
      icon: resumeIcon,
      disabled: false,
      variant: 'success',
      callback: props.resumeCallback,
      buttonWidth: backResumeButtonWidth,
      buttonHeight: footerButtonHeight,
      fontSize: backResumeFontSize,
      iconSize: footerButtonHeight,
      backgroundColor: null
    }
  }

  /**
   * Get the footer text and button to render in footer.
   *
   * @param {object} config - a single nested object with properties of footer buttons
   *
   * @returns {JSX.Element} the footer text and button
   */
  const renderFooterButton = useCallback(
    (config) => {
      return (
        <>
          <Row className='justify-content-center' style={{ width: '100%' }}>
            <Button
              variant={config.variant}
              disabled={config.disabled}
              onClick={config.callback}
              style={{
                backgroundColor: config.backgroundColor,
                border: 'none',
                marginTop: footerTopBottomMargin,
                marginLeft: footerLeftRightMargin,
                marginRight: footerLeftRightMargin,
                marginBottom: footerTopBottomMargin,
                width: config.buttonWidth,
                height: config.buttonHeight,
                justifyContent: 'center',
                alignItems: 'center',
                '--bs-btn-padding-y': '0rem',
                '--bs-btn-padding-x': '0rem'
              }}
            >
              <View
                style={{
                  flex: 1,
                  flexDirection: 'row',
                  justifyContent: 'center',
                  alignItems: 'center',
                  width: '100%',
                  height: '100%'
                }}
              >
                <View style={{ flex: 1, justifyContent: 'center', alignItems: 'flex-end', width: '100%', height: '100%' }}>
                  <p
                    className='transitionMessage'
                    style={{
                      marginBottom: '0',
                      fontSize: config.fontSize,
                      color: 'black',
                      fontWeight: 'bold',
                      padding: '0',
                      textAlign: 'right'
                    }}
                  >
                    {config.text}
                  </p>
                </View>
                <View style={{ flex: 1, justifyContent: 'center', alignItems: 'flex-start', width: '100%', height: '100%' }}>
                  <div style={{ textAlign: 'left' }}>
                    <img
                      style={{
                        width: '100%',
                        height: config.iconSize
                      }}
                      src={config.icon}
                      alt='icon_img'
                    />
                  </div>
                </View>
              </View>
            </Button>
          </Row>
        </>
      )
    },
    [footerLeftRightMargin, footerTopBottomMargin]
  )

  // Render the component
  return (
    <View style={{ wdith: '100%' }}>
      <MDBFooter bgColor='dark' className='text-center text-lg-left' style={{ width: '100vw' }}>
        <div className='text-center' style={{ backgroundColor: 'rgba(0, 0, 0, 0.2)', paddingBottom: '5px', paddingTop: '5px' }}>
          <View style={{ flex: 1, flexDirection: 'row', justifyContent: 'center', alignItems: 'center', width: '100%' }}>
            {props.paused ? (
              <>
                <View
                  style={{ flex: 1, flexDirection: 'row', justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}
                >
                  {props.backCallback ? renderFooterButton(buttonConfig.back) : <></>}
                </View>
                <View
                  style={{ flex: 1, flexDirection: 'row', justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}
                >
                  {props.resumeCallback ? renderFooterButton(buttonConfig.resume) : <></>}
                </View>
              </>
            ) : (
              renderFooterButton(buttonConfig.pause)
            )}
          </View>
        </div>
      </MDBFooter>
    </View>
  )
}
Footer.propTypes = {
  mealState: PropTypes.string.isRequired,
  paused: PropTypes.bool.isRequired,
  pauseCallback: PropTypes.func.isRequired,
  // If any of the below two are null, the Footer won't render that button
  resumeCallback: PropTypes.func,
  backCallback: PropTypes.func,
  backMealState: PropTypes.string
}

export default Footer
