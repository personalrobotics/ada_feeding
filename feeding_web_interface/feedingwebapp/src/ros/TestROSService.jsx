// React imports
import React, { useState } from 'react'

// Local imports
import { connectToROS, createROSService, createROSServiceRequest } from './ros_helpers'

/**
 * The TestROSService component demonstrates the functionality of calling a ROS
 * service. It uses the sample ROS service, `ReverseString`, which is defined in
 * `feeding_web_app_ros2_test/reverse_string_service.py`.
 */
function TestROSService() {
  // The defaults to use on this page
  let serviceName = '/reverse_string'
  let defaultInput = 'Hello World!'

  // Configure local state, which should contain any data that we want to
  // persist across re-renderings (and/or any data that, when changed, should
  // trigger re-renderings)
  let [recvData, setRecvData] = useState('No response received yet.')

  // Connect to ROS, if not already connected
  let { ros } = connectToROS()

  // Create the Service
  let reverseStringService = createROSService(ros, serviceName, 'feeding_web_app_ros2_msgs/ReverseString')

  // Callback function for when the user clicks the "Call" button
  function callService(event) {
    // Prevent the browser from reloading the page
    event.preventDefault()

    // Get the message from the form
    let input = event.target.input.value

    // Create a service request
    let request = createROSServiceRequest({ input: input })

    // Create the callback function
    let callback = function (response) {
      setRecvData(response.reversed)
    }

    // Publish the message
    reverseStringService.callService(request, callback)
  }

  // Render the component
  return (
    <div>
      {/**
       * Allow users to call the `/reverse_string` service
       */}
      <h4>Call the &apos;reverse_string&apos; Service:</h4>
      <form method='post' onSubmit={callService}>
        Input: <input type='text' name='input' defaultValue={defaultInput} />
        <button type='submit'>Call</button>
        <br />
        <br />
        <h4>Response:</h4>
        {recvData}
      </form>
    </div>
  )
}

export default TestROSService
