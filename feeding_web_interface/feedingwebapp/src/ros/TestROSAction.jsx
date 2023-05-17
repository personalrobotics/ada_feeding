// React imports
import React, { useState } from 'react'

// Local imports
import { useROS, createROSActionClient, callROSAction, cancelROSAction } from './ros_helpers'

/**
 * The TestROSAction component demonstrates the functionality of calling a ROS
 * action. It uses the sample ROS action, `SortByCharcterFrequency`, which is
 * defined in `feeding_web_app_ros2_test/sort_by_character_frequency_action.py`.
 */
function TestROSAction() {
  // The defaults to use on this page
  let actionName = '/sort_by_character_frequency'
  let defaultInput = 'Hello World!'

  // Connect to ROS, if not already connected
  let { ros } = useROS()

  // Configure local state, which should contain any data that we want to
  // persist across re-renderings (and/or any data that, when changed, should
  // trigger re-renderings)
  let [recvFeedback, setRecvFeedback] = useState('No feedback received yet.')
  let [recvResponse, setRecvResponse] = useState('No response received yet.')
  // Create the Service
  let sortByCharacterFrequencyAction = useState(
    createROSActionClient(ros, actionName, 'feeding_web_app_ros2_msgs/action/SortByCharacterFrequency')
  )[0]

  // Callback function for when the user clicks the "Call" button
  function callAction(event) {
    console.log('Calling action...')
    // Prevent the browser from reloading the page
    event.preventDefault()

    // Reset the feedback display
    setRecvFeedback('No feedback received yet.')

    // Get the message from the form
    let input = event.target.input.value

    // Create the feedback callback function
    let feedbackCallback = function (feedback) {
      console.log('Feedback: ', feedback)
      setRecvFeedback(feedback.values.feedback.progress)
    }

    // Create the feedback callback function
    let responseCallback = function (response) {
      console.log('Response: ', response)
      let displayStr = response.response_type.concat(': ', response.values.result)
      setRecvResponse(displayStr)
    }

    // Call the action
    callROSAction(sortByCharacterFrequencyAction, { input: input }, feedbackCallback, responseCallback)
  }

  // Callback function for when the user clicks the "Cancel" button
  function cancelAction() {
    console.log('Canceling action...')
    // Cancel the action
    cancelROSAction(sortByCharacterFrequencyAction)
  }

  // Render the component
  return (
    <div>
      {/**
       * Allow users to call the `/sort_by_character_frequency` service
       */}
      <h4>Call the &apos;sort_by_character_frequency&apos; Action:</h4>
      <form method='post' onSubmit={callAction}>
        Input: <input type='text' name='input' defaultValue={defaultInput} />
        <button type='submit'>Call</button>
        <br />
        <br />
        <h4>Feedback:</h4>
        {recvFeedback}
        <h4>Response:</h4>
        {recvResponse}
        <br />
      </form>
      <button type='cancel' onClick={cancelAction}>
        Cancel
      </button>
    </div>
  )
}

export default TestROSAction
