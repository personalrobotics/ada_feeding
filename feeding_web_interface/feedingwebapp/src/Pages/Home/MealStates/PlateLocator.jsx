// React Imports
import React, { useCallback, useMemo, useRef } from 'react'
import Button from 'react-bootstrap/Button'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'

// Local Imports
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import { convertRemToPixels } from '../../../helpers'
import { Col, Row, Container } from 'react-bootstrap'
import VideoFeed from '../VideoFeed'

/**
 * The PlateLocator component appears if the user decides to adjust the position
 * of the fork above the plate before selecting a bite. This component enables
 * the user to teleoperate the robot with Cartesian Control until the plate
 * is satisfactorily in view.
 */
const PlateLocator = (props) => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  // Get robot motion flag for plate locator
  const setTeleopIsMoving = useGlobalState((state) => state.setTeleopIsMoving)

  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'

  // Variables to render the VideoFeed
  const videoParentRef = useRef(null)
  // Margin for the video feed and between the mask buttons. Note this cannot
  // be re-defined per render, otherwise it messes up re-rendering order upon
  // resize in VideoFeed.
  const margin = useMemo(() => convertRemToPixels(1), [])

  // text font size
  let textFontSize = isPortrait ? '2vh' : '2vw'
  // done button width
  let doneButtonWidth = '47vw'
  // button height
  let buttonHeight = isPortrait ? '6vh' : '12vh'
  // arrow button width
  let arrowButtonWidth = '13vw'

  /**
   * Callback function for when the user presses one of the buttons to teleop
   * the robot.
   *
   * TODO: Implement this when ROS is connected to the robot!
   */
  const cartesianControlCommandReceived = useCallback(
    (event) => {
      let direction = event.target.value
      setTeleopIsMoving(true)
      console.log('cartesianControlCommandReceived', direction)
    },
    [setTeleopIsMoving]
  )

  /**
   * Callback function for when the user indicates that they are done
   * teleoperating the robot.
   */
  const doneClicked = useCallback(() => {
    console.log('doneClicked')
    setTeleopIsMoving(false)
    setMealState(MEAL_STATE.U_BiteSelection)
  }, [setMealState, setTeleopIsMoving])

  /**
   * Get the done button to click when locating plate is done.
   *
   * @returns {JSX.Element} the done button
   */
  const doneButton = useCallback(() => {
    return (
      <center>
        <Button variant='success' onClick={doneClicked} style={{ width: doneButtonWidth, height: buttonHeight, fontSize: textFontSize }}>
          ✅ Done
        </Button>
      </center>
    )
  }, [doneClicked, textFontSize, buttonHeight, doneButtonWidth])

  /**
   * An array of directional buttons for the user to teleoperate the robot, and a
   * button for the user to indicate that they are done teleoperating the robot.
   *
   * @returns {JSX.Element} the directional arrow buttons
   */
  const directionalArrows = useCallback(() => {
    return (
      <>
        <Container fluid>
          <Row className='justify-content-center'>
            <Button
              onClick={cartesianControlCommandReceived}
              style={{
                fontSize: textFontSize,
                alignItems: 'right',
                justifyContent: 'center',
                width: arrowButtonWidth,
                height: buttonHeight,
                marginTop: margin
              }}
              value='forward'
              variant='primary'
            >
              ⬆
            </Button>
          </Row>
          <Row className='justify-content-center' style={{ '--bs-gutter-x': '0rem' }} xs='auto'>
            <Col>
              <center>
                <Button
                  onClick={cartesianControlCommandReceived}
                  style={{
                    fontSize: textFontSize,
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: arrowButtonWidth,
                    height: buttonHeight
                  }}
                  value='left'
                  variant='primary'
                >
                  ⬅
                </Button>
              </center>
            </Col>
            <Col style={{ width: '55px' }}></Col>
            <Col>
              <center>
                <Button
                  onClick={cartesianControlCommandReceived}
                  style={{
                    fontSize: textFontSize,
                    alignItems: 'left',
                    justifyContent: 'center',
                    width: arrowButtonWidth,
                    height: buttonHeight
                  }}
                  value='right'
                  variant='primary'
                >
                  ➡
                </Button>
              </center>
            </Col>
          </Row>
          <Row className='justify-content-center'>
            <Button
              onClick={cartesianControlCommandReceived}
              style={{
                fontSize: textFontSize,
                alignItems: 'center',
                justifyContent: 'center',
                width: arrowButtonWidth,
                height: buttonHeight,
                marginBottom: margin
              }}
              value='back'
              variant='primary'
            >
              ⬇
            </Button>
          </Row>
        </Container>
      </>
    )
  }, [cartesianControlCommandReceived, arrowButtonWidth, buttonHeight, textFontSize, margin])

  // Render the component
  return (
    <View style={{ flex: 'auto', flexDirection: dimension, justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}>
      <View ref={videoParentRef} style={{ flex: 5, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
        <VideoFeed
          webVideoServerURL={props.webVideoServerURL}
          parent={videoParentRef}
          marginTop={margin}
          marginBottom={margin}
          marginLeft={margin}
          marginRight={margin}
        />
      </View>
      <View style={{ flex: 5, alignItems: 'center', justifyContent: 'center' }}>
        {directionalArrows()}
        {doneButton()}
      </View>
    </View>
  )
}
PlateLocator.propTypes = {
  // The URL of the web video server
  webVideoServerURL: PropTypes.string.isRequired
}

export default PlateLocator
