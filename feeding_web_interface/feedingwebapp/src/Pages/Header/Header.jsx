/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React, { useCallback, useEffect, useState } from 'react'
import PropTypes from 'prop-types'
// The NavBar is the navigation toolbar at the top
import Navbar from 'react-bootstrap/Navbar'
import Nav from 'react-bootstrap/Nav'
import { useMediaQuery } from 'react-responsive'
// Toast generates a temporary pop-up with a timeout.
import { ToastContainer, toast } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
// ROS imports
import { useROS } from '../../ros/ros_helpers'
// Local imports
import { REGULAR_CONTAINER_ID, ROS_CHECK_INTERVAL_MS } from '../Constants'
import { useGlobalState, APP_PAGE, NON_MOVING_STATES } from '../GlobalState'
import InfoModal from './InfoModal'

/**
 * The Header component consists of the navigation bar (which has buttons Home,
 * Settings, Lock and Robot Connection Icon and Video). Live video view is toggled on and off by
 * clicking "Video", and the ToastContainer popup that specifies when the user
 * cannot click Settings.
 */
const Header = (props) => {
  // Create a local state variable to toggle on/off the video
  // TODO: Since this local state variable is in the header, the InfoModal
  // continues showing even if the state changes. Is this desirable? Perhaps
  // it should close if the state changes?
  const [infoModalShow, setInfoModalShow] = useState(false)
  // useROS gives us access to functions to configure and interact with ROS.
  let { ros } = useROS()
  const [isConnected, setIsConncected] = useState(ros.isConnected)
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Sizes of header elements (fontSize, width, height)
  let textFontSize = isPortrait ? '3vh' : '6vh'
  let lockImageHeight = isPortrait ? '4vh' : '8vh'
  let lockImageWidth = '4vw'

  // Check ROS connection every ROS_CHECK_INTERVAL_MS milliseconds
  useEffect(() => {
    const interval = setInterval(() => {
      setIsConncected(ros.isConnected)
    }, ROS_CHECK_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [ros, setIsConncected])

  // Get the relevant global state variables
  const mealState = useGlobalState((state) => state.mealState)
  const inNonMovingState = useGlobalState((state) => state.inNonMovingState)
  const paused = useGlobalState((state) => state.paused)
  const setAppPage = useGlobalState((state) => state.setAppPage)

  /**
   * When the Home button in the header is clicked, return to the Home page.
   */
  const homeClicked = useCallback(() => {
    setAppPage(APP_PAGE.Home)
  }, [setAppPage])

  /**
   * When the Settings button in the header is clicked, if the meal has not yet
   * started, take the user to the settings menu. Else, ask them to complete
   * or terminate the meal because modifying settings.
   */
  const settingsClicked = useCallback(() => {
    // Both checks are necessary because some states that are nominally non-moving
    // could be in an auto-continue state, and some states that are nominally moving
    // could be paused.
    if ((inNonMovingState && NON_MOVING_STATES.has(mealState)) || (!NON_MOVING_STATES.has(mealState) && paused)) {
      setAppPage(APP_PAGE.Settings)
    } else {
      toast.info('Wait for robot motion to complete before accessing Settings.', {
        containerId: REGULAR_CONTAINER_ID,
        toastId: 'noSettings'
      })
    }
  }, [inNonMovingState, mealState, paused, setAppPage])

  // Render the component. The NavBar will stay fixed even as we vertically scroll.
  return (
    <>
      {/**
       * The ToastContainer is an alert that pops up on the top of the screen
       * and has a timeout.
       */}
      <ToastContainer style={{ fontSize: textFontSize, zIndex: 9999 }} containerId={REGULAR_CONTAINER_ID} enableMultiContainer={true} />
      {/**
       * The NavBar has two elements, Home and Settings, on the left side and three
       * elements, Lock, Robot Connection Icon and VideoVideo, on the right side.
       * An image showing the connection status
       * of the robot is placed in between Lock and Video.
       */}
      <Navbar
        collapseOnSelect
        expand='lg'
        bg='dark'
        variant='dark'
        sticky='top'
        style={{ width: '100vw', '--bs-navbar-padding-x': '0rem', '--bs-navbar-padding-y': '0rem' }}
      >
        <Navbar
          id='responsive-navbar-nav'
          bg='dark'
          variant='dark'
          style={{ '--bs-navbar-padding-x': '0rem', '--bs-navbar-padding-y': '0.2rem' }}
        >
          <Nav className='me-auto'>
            <Nav.Link
              onClick={homeClicked}
              className='text-dark bg-info rounded mx-1 btn-lg btn-huge p-2'
              style={{ fontSize: textFontSize }}
            >
              Home
            </Nav.Link>
            <Nav.Link
              onClick={settingsClicked}
              className='text-dark bg-info rounded mx-1 btn-lg btn-huge p-2'
              style={{ fontSize: textFontSize }}
            >
              Settings
            </Nav.Link>
          </Nav>
          {inNonMovingState ? (
            <Nav>
              <Nav.Link
                className='text-dark rounded mx-1 btn-lg btn-huge p-2'
                style={{ fontSize: textFontSize, backgroundColor: '#dc3545' }}
              >
                <img
                  style={{ width: lockImageWidth, height: lockImageHeight }}
                  src='/robot_state_imgs/lock_icon_image.svg'
                  alt='lock_icon_img'
                  className='center'
                />
              </Nav.Link>
            </Nav>
          ) : (
            <></>
          )}
          {isConnected ? (
            <Nav>
              <Nav.Link className='text-dark rounded mx-1 btn-lg btn-huge p-2' style={{ fontSize: textFontSize, backgroundColor: 'green' }}>
                🔌
              </Nav.Link>
            </Nav>
          ) : (
            <Nav>
              <Nav.Link
                className='text-dark rounded mx-1 btn-lg btn-huge p-2'
                style={{ fontSize: textFontSize, backgroundColor: '#f0ad4e' }}
              >
                ⛔
              </Nav.Link>
            </Nav>
          )}
          <Nav>
            <Nav.Link
              onClick={() => setInfoModalShow(true)}
              className='text-dark bg-info rounded mx-1 btn-lg btn-huge p-2'
              style={{ fontSize: textFontSize }}
            >
              Info
            </Nav.Link>
          </Nav>
        </Navbar>
      </Navbar>
      {/**
       * The InfoModal toggles on and off with the Video button and shows the
       * robot's live camera feed.
       */}
      <InfoModal show={infoModalShow} onHide={() => setInfoModalShow(false)} webrtcURL={props.webrtcURL} />
    </>
  )
}
Header.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default Header
