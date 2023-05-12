// React Imports
import React, { useState, useCallback, useEffect } from 'react'
import Button from 'react-bootstrap/Button'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'
import Row from 'react-bootstrap/Row'

// Local Imports
import { connectToROS, createROSActionClient, callROSAction, cancelROSAction, destroyActionClient } from '../../../ros/ros_helpers'
import Footer from '../../Footer/Footer'
import '../Home.css'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import {
  ROS_ACTIONS_NAMES,
  MOTION_STATUS_SUCCESS,
  ROS_ACTION_STATUS_CANCEL_GOAL,
  ROS_ACTION_STATUS_EXECUTE,
  ROS_ACTION_STATUS_SUCCEED,
  ROS_ACTION_STATUS_ABORT,
  ROS_ACTION_STATUS_CANCELED
} from '../../Constants'

/**
 * The MovingAbovePlate component tells the user that the robot is currently
 * moving above the plate. It waits for the robot to complete the motion before
 * moving to the next state.
 *
 * @params {object} props - contains any properties passed to this Component
 */
const MovingAbovePlate = (props) => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)

  // Create a local state variable for whether the robot is paused and another
  // to store the status of the action
  const [paused, setPaused] = useState(false)
  // NOTE: We slightly abuse the ROS_ACTION_STATUS values in this local state
  // variable, by using it as a proxy for whether the robot is executing, has
  // succeeded, has been canceled, or has had an error. This is to avoid
  // creating an extra enum.
  const [actionStatus, setActionStatus] = useState({
    actionStatus: null
  })

  // Connect to ROS, if not already connected. Put this in local state to avoid
  // re-connecting upon every re-render.
  const ros = useState(connectToROS().ros)[0]

  // Create the ROS Action Client. This is created in local state to avoid
  // re-creating it upon every re-render.
  let { actionName, messageType } = ROS_ACTIONS_NAMES[MEAL_STATE.R_MovingAbovePlate]
  let moveAbovePlateAction = useState(createROSActionClient(ros, actionName, messageType))[0]

  /**
   * Callback function for when the action sends feedback. It updates the
   * actionStatus local state variable.
   *
   * @param {object} feedbackMsg - the feedback message sent by the action
   */
  const feedbackCallback = useCallback(
    (feedbackMsg) => {
      setActionStatus({
        actionStatus: ROS_ACTION_STATUS_EXECUTE,
        feedback: feedbackMsg.values.feedback
      })
    },
    [setActionStatus]
  )

  /**
   * Callback function for when the robot has finished moving above the plate.
   * It moves on to the next meal state.
   */
  const movingAbovePlateDone = useCallback(() => {
    console.log('movingAbovePlateDone')
    setMealState(MEAL_STATE.U_BiteSelection)
  }, [setMealState])

  /**
   * Callback function for when the action sends a response. It updates the
   * actionStatus local state variable and moves on to the next state if the
   * action succeeded.
   *
   * TODO: What should happen if the action succeeds, but a response is never
   * received e.g., due to a momentary networking issue? We should likely allow
   * the user to press a button to move on if a response hasn't been received
   * within n seconds of making the action call. Or allow the user to retry the
   * action call (maybe they would refresh the page anyway, which might be ok).
   */
  const responseCallback = useCallback(
    (response) => {
      if (response.response_type == 'result' && response.values.status == MOTION_STATUS_SUCCESS) {
        setActionStatus({
          actionStatus: ROS_ACTION_STATUS_SUCCEED
        })
        movingAbovePlateDone()
      } else {
        if (
          response.response_type == 'cancel' ||
          response.values == ROS_ACTION_STATUS_CANCEL_GOAL ||
          response.values == ROS_ACTION_STATUS_CANCELED
        ) {
          setActionStatus({
            actionStatus: ROS_ACTION_STATUS_CANCELED
          })
        } else {
          setActionStatus({
            actionStatus: ROS_ACTION_STATUS_ABORT
          })
        }
      }
    },
    [movingAbovePlateDone, setActionStatus]
  )

  /**
   * Callback function for when the pause button is pressed. It cancels the
   * action.
   */
  const pauseCallback = useCallback(() => {
    cancelROSAction(moveAbovePlateAction)
  }, [moveAbovePlateAction])

  /**
   * Function to call the ROS action. Note that every time this function
   * is called it re-registers callbacks, so typically callbacks should only
   * be passed the first time it is called.
   *
   * @param {function} feedbackCb - the callback function for when the action
   * sends feedback
   * @param {function} responseCb - the callback function for when the action
   * sends a response
   */
  const callMoveAbovePlate = useCallback(
    (feedbackCb, responseCb) => {
      if (!paused) {
        setActionStatus({
          actionStatus: ROS_ACTION_STATUS_EXECUTE
        })
        callROSAction(moveAbovePlateAction, {}, feedbackCb, responseCb)
      }
    },
    [moveAbovePlateAction, setActionStatus]
  )
  /**
   * Calls the action the first time this component is rendered, but not upon
   * any additional re-renders. See here for more details on how `useEffect`
   * achieves this goal: https://stackoverflow.com/a/69264685
   */
  useEffect(() => {
    callMoveAbovePlate(feedbackCallback, responseCallback)
    // In practice, because the values passed in in the second argument of
    // useEffect will not change on re-renders, this return statement will
    // only be called when the component unmounts.
    return () => {
      destroyActionClient(moveAbovePlateAction)
    }
  }, [callMoveAbovePlate, moveAbovePlateAction, feedbackCallback, responseCallback])

  /**
   * Callback function for when the resume button is pressed. It calls the
   * action once again.
   */
  const resumeCallback = useCallback(() => {
    callMoveAbovePlate(null, null) // don't re-register the callbacks
  }, [callMoveAbovePlate])

  /**
   * Get the action status text to render. Note that once Issue #22 is addressed,
   * this will likely no longer be necessary (since that issue focuses on
   * visually rendering robot progress).
   *
   * @returns {JSX.Element} the action status text to render
   */
  let actionStatusText = function () {
    switch (actionStatus.actionStatus) {
      case ROS_ACTION_STATUS_EXECUTE:
        if (actionStatus.feedback) {
          if (!actionStatus.feedback.is_planning) {
            let progress = 1 - actionStatus.feedback.motion_curr_distance / actionStatus.feedback.motion_initial_distance
            return (
              <>
                <h3>Robot is moving...</h3>
                <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Progress: {Math.round(progress * 100)}%</h3>
              </>
            )
          } else {
            let elapsed_time = actionStatus.feedback.planning_time.sec + actionStatus.feedback.planning_time.nanosec / 10 ** 9
            return (
              <>
                <h3>Robot is thinking...</h3>
                <h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Elapsed Time: {Math.round(elapsed_time * 100) / 100} sec</h3>
              </>
            )
          }
        } else {
          return <h3></h3>
        }
      case ROS_ACTION_STATUS_SUCCEED:
        return <h3>Robot has finished</h3>
      case ROS_ACTION_STATUS_ABORT:
        /**
         * TODO: Just displaying that the robot faced an error is not useful
         * to the user. We should think more carefully about what different
         * error cases might arise, and change the UI accordingly to instruct
         * users on how to troubleshoot/fix it.
         */
        return <h3>Robot encountered an error</h3>
      case ROS_ACTION_STATUS_CANCELED:
        return <h3>Robot is paused</h3>
      default:
        return <h3></h3>
    }
  }

  // Render the component
  return (
    <>
      {/* TODO: Consider vertically centering this element */}
      <Row className='justify-content-center mx-auto my-2 w-75'>
        <div>
          <h1 id={MEAL_STATE.R_MovingAbovePlate} className='waitingMsg'>
            Waiting for the robot to move above the plate...
          </h1>
          {props.debug ? (
            <Button variant='secondary' className='justify-content-center mx-2 mb-2' size='lg' onClick={movingAbovePlateDone}>
              Continue (Debug Mode)
            </Button>
          ) : (
            <></>
          )}
          <br />
          <br />
          <br />
          {/**
           * TODO (Issue #22): Instead of just displaying progress via text on
           * the page, we should visually show a progress bar. This will also
           * negate the need for the `actionStatusText` function.
           */}
          {actionStatusText()}
        </div>
      </Row>
      {/**
       * Display the footer with the Pause button. MoveAbovePlate has no back
       * button, because the user can go "forward" to any state with similar
       * ease as they could go "backwards" to the same state.
       */}
      <Footer
        pauseCallback={pauseCallback}
        backCallback={null}
        backMealState={null}
        resumeCallback={resumeCallback}
        paused={paused}
        setPaused={setPaused}
      />
    </>
  )
}
MovingAbovePlate.propTypes = {
  debug: PropTypes.bool.isRequired
}

export default MovingAbovePlate
