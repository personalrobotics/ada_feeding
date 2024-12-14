/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React from 'react'
import PropTypes from 'prop-types'

// Local imports
import { ROBOT_COMPRESSED_IMG_TOPICS } from '../Pages/Constants'
import VideoStream from './VideoStream'

/**
 * Renders all the video streams from the robot.
 *
 * NOTE: This page *must* be rendered on the robot, otherwise it will have
 * incredible lag.
 *
 * @param {string} webrtcURL - The URL of the webrtc signalling server.
 */
function RobotVideoStreams(props) {
  console.log('Rendering RobotVideoStreams', ROBOT_COMPRESSED_IMG_TOPICS)
  return (
    <>
      {ROBOT_COMPRESSED_IMG_TOPICS.map((topic, i) => (
        <VideoStream topic={topic} key={i} webrtcURL={props.webrtcURL} />
      ))}
    </>
  )
}
RobotVideoStreams.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default RobotVideoStreams
