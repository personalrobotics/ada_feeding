// React imports
import React from 'react'

// Local imports
import { useROS } from './ros_helpers'
import TestROSAction from './TestROSAction'
import TestROSPublish from './TestROSPublish'
import TestROSService from './TestROSService'
import TestROSSubscribe from './TestROSSubscribe'

/**
 * The TestROS component demonstrates the functionality of every implemented
 * function in `ros_helpers`. It is not part of the main app, and is merely meant
 * to be a reference for how to use the ROS helper functions.
 */
function TestROS() {
  // Connect to ROS, if not already connected
  let { isConnected } = useROS()

  // Render the component
  return (
    <div>
      {/**
       * Display whether the web app is connected to ROS.
       */}
      {isConnected ? (
        <div>
          <p className='connectedDiv' style={{ fontSize: '24px' }}>
            🔌 connected
          </p>
        </div>
      ) : (
        <div>
          <p className='notConnectedDiv' style={{ fontSize: '24px' }}>
            ⛔ not connected
          </p>
        </div>
      )}
      <hr />

      {/**
       * Allow users to create and publish to a topic
       */}
      <TestROSPublish />
      <hr />

      {/**
       * Allow users to subscribe to a topic and display its data
       */}
      <TestROSSubscribe />
      <hr />

      {/**
       * Allow users to call a ROS Service
       */}
      <TestROSService />
      <hr />

      {/**
       * Allow users to call a ROS Action
       */}
      <TestROSAction />
      <hr />
    </div>
  )
}

export default TestROS
