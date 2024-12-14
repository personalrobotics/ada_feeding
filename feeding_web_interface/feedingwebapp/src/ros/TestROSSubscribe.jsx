/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React, { useState } from 'react'

// Local imports
import { useROS, subscribeToROSTopic } from './ros_helpers'

/**
 * The TestROSSubscribe component demonstrates the functionality of subscribing
 * to a ROS topic.
 */
function TestROSSubscribe() {
  // The defaults to use on this page
  let defaultTopicName = 'test_topic'

  // Connect to ROS, if not already connected
  let { ros } = useROS()

  // Configure local state, which should contain any data that we want to
  // persist across re-renderings (and/or any data that, when changed, should
  // trigger re-renderings)
  let [recvData, setRecvData] = useState('No message received yet.')

  // Callback function for when the user clicks the "Subscribe" button
  function subscribeTopic(event) {
    // Prevent the browser from reloading the page
    event.preventDefault()

    // Get the topic name from the form
    let topicName = event.target.topicName.value

    // Create the callback function
    let callback = function (message) {
      setRecvData(message.data)
    }

    // Subscribe to the topic
    subscribeToROSTopic(ros, topicName, 'std_msgs/String', callback)
  }

  // Render the component
  return (
    <div>
      {/**
       * Allow users to subscribe to a topic and display its data
       */}
      <h4>Subscribe to a &apos;std_msgs/String&apos; Topic and Display Its Data:</h4>
      <form method='post' onSubmit={subscribeTopic}>
        Topic Name: <input type='text' name='topicName' defaultValue={defaultTopicName} />
        <button type='submit'>Subscribe</button>
        <br />
        <br />
        <h4>Received Data:</h4>
        {recvData}
      </form>
    </div>
  )
}

export default TestROSSubscribe
