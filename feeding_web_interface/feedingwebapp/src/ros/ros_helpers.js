/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import { useRos as useRosReact } from 'rosreact'
import ROSLIB from 'roslib'

/**
 * Connects to ROS, if not already connected.
 *
 * @returns {boolean} isConnected True if ROS is connected, false otherwise.
 * @returns {object} ros The ROSLIB.Ros object.
 */
export function useROS() {
  let ros = useRosReact()
  return { isConnected: ros.isConnected, ros }
}

/**
 * Creates a ROS message.
 *
 * @param {object} data An object containing the exact attributes and types expected by the ROS message.
 *
 * @returns {object} The ROSLIB.Message
 */
export function createROSMessage(data) {
  return new ROSLIB.Message(data)
}

/**
 * Creates and advertises a ROS topic.
 *
 * @param {object} ros The ROSLIB.Ros object.
 * @param {string} topicName The name of the topic to create.
 * @param {string} topicType The type of the topic to create.
 *
 * @returns {object} The ROSLIB.Topic, or null if ROS is not connected.
 */
export function createROSTopic(ros, topicName, topicType) {
  if (ros === null) {
    console.log('ROS is not connected')
    return null
  }
  let topic = new ROSLIB.Topic({
    ros: ros,
    name: topicName,
    messageType: topicType
  })
  topic.advertise()
  return topic
}

/**
 * Subscribes to a ROS topic.
 *
 * @param {object} ros The ROSLIB.Ros object.
 * @param {string} topicName The name of the topic to create.
 * @param {string} topicType The type of the topic to create.
 * @param {function} callback The callback function to call when a message is
 *                   received.
 * @returns {object} The ROSLIB.Topic, or null if ROS is not connected.
 */
export function subscribeToROSTopic(ros, topicName, topicType, callback) {
  if (ros === null) {
    console.log('ROS is not connected')
    return null
  }
  let topic = new ROSLIB.Topic({
    ros: ros,
    name: topicName,
    messageType: topicType
  })
  topic.subscribe(callback)
  return topic
}

/**
 * Unsubscribe from a ROS topic.
 *
 * @param {object} topic The ROSLIB.Topic.
 * @param {function} callback The callback function to unsubscribe.
 */
export function unsubscribeFromROSTopic(topic, callback) {
  topic.unsubscribe(callback)
}

/**
 * Create a ROS Service.
 *
 * @param {object} ros The ROSLIB.Ros object.
 * @param {string} serviceName The name of the service to create.
 * @param {string} serviceType The type of the service to create.
 *
 * @returns {object} The ROSLIB.Service, or null if ROS is not connected.
 */
export function createROSService(ros, serviceName, serviceType) {
  if (ros === null) {
    console.log('ROS is not connected')
    return null
  }
  let service = new ROSLIB.Service({
    ros: ros,
    name: serviceName,
    serviceType: serviceType
  })
  return service
}

/**
 * Creates a ROS Service Request.
 *
 * @param {object} data An object containing the exact attributes and types
 *                 expected by the ROS Service.
 *
 * @returns {object} The ROSLIB.ServiceRequest
 */
export function createROSServiceRequest(data) {
  return new ROSLIB.ServiceRequest(data)
}

// TODO: Should we add a `destroyService` function like we have for the action client?
// The difference is that roslibjs explicitly provides a function to destroy the action
// client, but only provides a function to unadvertise a service client.

/**
 * Create a ROS Action Client.
 *
 * @param {object} ros The ROSLIB.Ros object.
 * @param {string} serverName The name of the action server to call.
 * @param {string} actionType The type of the action the server takes.
 *
 * @returns {object} The ROSLIB.ActionHandle, or null if ROS is not connected.
 */
export function createROSActionClient(ros, serverName, actionType) {
  if (ros === null) {
    console.log('ROS is not connected')
    return null
  }
  let actionClient = new ROSLIB.ActionHandle({
    ros: ros,
    name: serverName,
    actionType: actionType
  })
  return actionClient
}

/**
 * Calls a ROS Action.
 *
 * @param {object} actionClient The ROSLIB.ActionHandle object.
 * @param {object} goal An object containing the exact attributes and types
 *                      expected by the ROS Action.
 * @param {function} feedbackCallback The callback function to call when
 *                   feedback is received.
 * @param {function} resultCallback The callback function to call when a
 *                                  result is received.
 */
export function callROSAction(actionClient, goal, feedbackCallback, resultCallback) {
  // TODO: After the callback is executed, it should be removed from the action
  // server's callback list to prevent multiple callbacks from being executed in
  // further calls.
  actionClient.createClient(goal, resultCallback, feedbackCallback)
  console.log('Called ROS Action')
}

/**
 * Cancels all the goals a ROS Action is currently executing.
 *
 * @param {object} actionClient The ROSLIB.ActionHandle object.
 */
export function cancelROSAction(actionClient) {
  actionClient.cancelGoal()
}

/**
 * Destroys an action client on the rosbridge end (e.g., so when the user returns
 * to the same state, it doesn't create a second client for the same action.)
 *
 * @param {object} actionClient The ROSLIB.ActionHandle object.
 */
export function destroyActionClient(actionClient) {
  actionClient.destroyClient()
}

/**
 * Takes in an object of type ParameterValue and returns the actual parameter
 * value. See the message definition for more details:
 * https://github.com/ros2/rcl_interfaces/blob/rolling/rcl_interfaces/msg/ParameterValue.msg
 *
 * @param {object} parameter an object of message type ParameterValue
 */
export function getValueFromParameter(parameter) {
  switch (parameter.type) {
    case 1: // bool
      return parameter.bool_value
    case 2: // integer
      return parameter.integer_value
    case 3: // double
      return parameter.double_value
    case 4: // string
      return parameter.string_value
    case 5: // byte array
      return parameter.byte_array_value
    case 6: // bool array
      return parameter.bool_array_value
    case 7: // integer array
      return parameter.integer_array_value
    case 8: // double array
      return parameter.double_array_value
    case 9: // string array
      return parameter.string_array_value
    default: // not set
      return null
  }
}

/**
 * Takes in a value in Javascript and returns a ParameterValue object. See the
 * message definition for more details:
 * https://github.com/ros2/rcl_interfaces/blob/rolling/rcl_interfaces/msg/ParameterValue.msg
 *
 * @param {any} value The value to convert to a ParameterValue object.
 * @param {number} typeOverride The type to treat the value as. If null, the
 *                              type will be inferred from the value.
 */
export function getParameterFromValue(value, typeOverride = null) {
  let parameter = new ROSLIB.Message({
    type: 0
  })
  if (typeOverride === 1 || typeof value === 'boolean') {
    parameter.bool_value = value
    parameter.type = 1
  } else if (typeOverride === 3 || typeof value === 'number') {
    parameter.double_value = value
    parameter.type = 3
  } else if (typeOverride === 2 || Number.isInteger(value)) {
    parameter.integer_value = value
    parameter.type = 2
  } else if (typeOverride === 4 || typeof value === 'string') {
    parameter.string_value = value
    parameter.type = 4
  } else if (typeOverride === 5 || (Array.isArray(value) && value.length > 0 && value[0] instanceof Uint8Array)) {
    parameter.byte_array_value = value
    parameter.type = 5
  } else if (typeOverride === 6 || (Array.isArray(value) && value.length > 0 && typeof value[0] === 'boolean')) {
    parameter.bool_array_value = value
    parameter.type = 6
  } else if (typeOverride === 8 || (Array.isArray(value) && value.length > 0 && typeof value[0] === 'number')) {
    parameter.double_array_value = value
    parameter.type = 8
  } else if (typeOverride === 7 || (Array.isArray(value) && value.length > 0 && Number.isInteger(value[0]))) {
    parameter.integer_array_value = value
    parameter.type = 7
  } else if (typeOverride === 9 || (Array.isArray(value) && value.length > 0 && typeof value[0] === 'string')) {
    parameter.string_array_value = value
    parameter.type = 9
  }

  if (parameter.type === 0) {
    console.log('Could not determine type of value', value)
  }

  return parameter
}
