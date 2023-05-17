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
  actionClient.createClient(goal, resultCallback, feedbackCallback)
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
