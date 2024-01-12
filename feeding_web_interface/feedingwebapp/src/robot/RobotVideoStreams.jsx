// React imports
import React from 'react'
import PropTypes from 'prop-types'

// Local imports
import { ROBOT_COMPRESSED_IMG_TOPICS } from '../Pages/Constants'
import VideoStream from './VideoStream'

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
