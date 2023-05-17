// React imports
import React, { useState } from 'react'

// Local imports
import { useROS, createROSTopic, createROSMessage } from './ros_helpers'

/**
 * The TestROSPublish component demonstrates the functionality of creating and
 * publishing to a ROS topic.
 */
function TestROSPublish() {
  // The defaults to use on this page
  let defaultTopicName = 'test_topic'
  let defaultMessage = 'Hello World!'

  // Connect to ROS, if not already connected
  let { ros } = useROS()

  // Configure local state, which should contain any data that we want to
  // persist across re-renderings (and/or any data that, when changed, should
  // trigger re-renderings)
  let [topic, setTopic] = useState(createROSTopic(ros, defaultTopicName, 'std_msgs/String'))

  // Callback function for when the user clicks the "Create Topic" button
  function createTopic(event) {
    // Prevent the browser from reloading the page
    event.preventDefault()

    // Get the topic name from the form
    let topicName = event.target.topicName.value

    // Create the topic
    setTopic(createROSTopic(ros, topicName, 'std_msgs/String'))
  }

  // Callback function for when the user clicks the "Publish" button
  function publishString(event) {
    // Prevent the browser from reloading the page
    event.preventDefault()

    // Get the message from the form
    let message = event.target.message.value

    // Publish the message
    topic.publish(createROSMessage({ data: message }))
  }

  // Render the component
  return (
    <div>
      {/**
       * Allow users to create and publish to a topic
       */}
      <h4>Create &apos;std_msgs/String&apos; Topic:</h4>
      <form method='post' onSubmit={createTopic}>
        Topic Name: <input type='text' name='topicName' defaultValue={defaultTopicName} />
        <button type='submit'>Create Topic</button>
      </form>
      <br />
      <h4>Publish to the Above Topic:</h4>
      <form method='post' onSubmit={publishString}>
        Message: <input type='text' name='message' defaultValue={defaultMessage} />
        <button type='submit'>Publish</button>
      </form>
    </div>
  )
}

export default TestROSPublish
