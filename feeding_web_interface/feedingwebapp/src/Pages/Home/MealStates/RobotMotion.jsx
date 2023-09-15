// React Imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMediaQuery } from 'react-responsive'
import Button from 'react-bootstrap/Button'
import { View } from 'react-native'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'
// Local Imports
import { useROS, createROSActionClient, callROSAction, cancelROSAction, destroyActionClient } from '../../../ros/ros_helpers'
import Footer from '../../Footer/Footer'
import CircleProgressBar from './CircleProgressBar'
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
 * The RobotMotion component is a generic component that tells the user the
 * robot is moving (with customizable text), calls the ROS action corresponding
 * to that motion, displays the progress, and moves to the next state when the
 * action is completed.
 *
 * @param {boolean} debug - whether to run it in debug mode (e.g., if you aren't
 *        simulatenously running the robot) or not
 * @param {string} mealState - the meal state corresponding with the motion the
 *         robot is executing
 * @param {string} nextMealState - the meal state to transition to once the
 *        robot finishes executing
 * @param {object} actionInput - the input to provide to the ROS action
 * @param {string} waitingText - the static text to display while the robot is
 *        executing the action
 */
const RobotMotion = (props) => {
  /**
   * NOTE: We slightly abuse the ROS_ACTION_STATUS values in this local state
   * variable, by using it as a proxy for whether the robot is executing, has
   * succeeded, has been canceled, or has had an error. This is to avoid
   * creating an extra enum.
   */
  const [actionStatus, setActionStatus] = useState({
    actionStatus: null
  })

  // Get the relevant global variables
  const mealState = useGlobalState((state) => state.mealState)
  const setMealState = useGlobalState((state) => state.setMealState)
  const paused = useGlobalState((state) => state.paused)
  const setPaused = useGlobalState((state) => state.setPaused)

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })

  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'
  // Waiting text font size
  let waitingTextFontSize = isPortrait ? '4.5vh' : '9vh'
  // Motion text font size
  let motionTextFontSize = isPortrait ? '3vh' : '6vh'

  /**
   * Create the ROS Action Client. This is re-created every time props.mealState
   * changes. Note that we use props here because there is sometimes a race
   * condition where the mealState has changed but this component is still
   * rendering, so we need to use the meal state intended for this component,
   * even if it momentarily differs from the global mealState.
   */
  let robotMotionAction = useMemo(() => {
    let { actionName, messageType } = ROS_ACTIONS_NAMES[props.mealState]
    return createROSActionClient(ros.current, actionName, messageType)
  }, [props.mealState])

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
   * Callback function for when the robot has finished moving to its staging
   * location.
   */
  const robotMotionDone = useCallback(() => {
    console.log('robotMotionDone')
    setMealState(props.nextMealState)
  }, [setMealState, props.nextMealState])

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
      if (response.response_type === 'result' && response.values.status === MOTION_STATUS_SUCCESS) {
        setActionStatus({
          actionStatus: ROS_ACTION_STATUS_SUCCEED
        })
        robotMotionDone()
      } else {
        if (
          response.response_type === 'cancel' ||
          response.values === ROS_ACTION_STATUS_CANCEL_GOAL ||
          response.values === ROS_ACTION_STATUS_CANCELED
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
    [robotMotionDone, setActionStatus]
  )

  /**
   * Callback function for when the pause button is pressed. It cancels the
   * action.
   */
  const pauseCallback = useCallback(() => {
    setPaused(true)
    cancelROSAction(robotMotionAction)
  }, [robotMotionAction, setPaused])

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
  const callRobotMotionAction = useCallback(
    (feedbackCb, responseCb) => {
      if (!paused) {
        setActionStatus({
          actionStatus: ROS_ACTION_STATUS_EXECUTE
        })
        callROSAction(robotMotionAction, props.actionInput, feedbackCb, responseCb)
      }
    },
    [robotMotionAction, paused, props.actionInput, setActionStatus]
  )

  /**
   * Calls the action the first time this component is rendered, but not upon
   * any additional re-renders. See here for more details on how `useEffect`
   * achieves this goal: https://stackoverflow.com/a/69264685
   */
  useEffect(() => {
    callRobotMotionAction(feedbackCallback, responseCallback)
    /**
     * In practice, because the values passed in in the second argument of
     * useEffect will not change on re-renders, this return statement will
     * only be called when the component unmounts.
     */
    return () => {
      destroyActionClient(robotMotionAction)
    }
  }, [callRobotMotionAction, robotMotionAction, feedbackCallback, responseCallback])

  /**
   * Callback function for when the resume button is pressed. It calls the
   * action once again.
   *
   * Note that BiteAcquistiion won't have a "Resume" button, since if the robot
   * has already touched the food then it may have shifted, which means the
   * previously-selected food mask may no longer be a valid goal.
   */
  const resumeCallback = useCallback(() => {
    setPaused(false)
    callRobotMotionAction(null, null) // don't re-register the callbacks
  }, [callRobotMotionAction, setPaused])

  /**
   * Callback function for when the back button is clicked. Regardless of the
   * state, all pressed of "back" will revert to the "Moving Above Plate" state.
   * - BiteAcquisition: In this case, pressing "back" should let the user
   *   reselect the bite, which requires the robot to move above plate.
   * - MoveToRestingPostion: In this case, pressing "back" should move the
   *   robot back to the plate. Although the user may not always want to
   *   reselect the bite, from `BiteSelection` they have the option to skip
   *   BiteAcquisition and move straight to resting positon (when they are ready).
   * - MoveToMouth: In this case, pressing "back" should move the
   *   robot back to the resting positon.
   * - StowingArm: In this case, if the user presses back they likely want to
   *   eat another bite, hence moving above the plate makes sense.
   * - MovingAbovePlate: Although the user may want to press "back" to move
   *   the robot to the mouth, they can also go forward to
   *   BiteSelection and then move the robot to the mouth location.
   *   Hence, in this case we don't have a "back" button.
   */
  const backMealState = useRef(MEAL_STATE.R_MovingAbovePlate)
  useEffect(() => {
    if (mealState === MEAL_STATE.R_MovingToMouth) {
      backMealState.current = MEAL_STATE.R_MovingToRestingPosition
    } else {
      backMealState.current = MEAL_STATE.R_MovingAbovePlate
    }
  }, [mealState, backMealState])
  const backCallback = useCallback(() => {
    setPaused(false)
    setMealState(backMealState.current)
  }, [setPaused, setMealState, backMealState])

  /**
   * Get the action status text and progress bar or blank view to render.
   *
   * @param {flexSizeOuter} - flexbox percentage for parent element rendering everything
   * @param {flexSizeTextInner} - flexbox percentage for child element rendering text
   * @param {flexSizeVisualInner} - flexbox percentage for child element rendering visual
   * @param {text} - action status text
   * @param {showTime} - indicates if elapsed time needs to be shown
   * @param {time} - calculated elapsed time, 0 if time not available
   * @param {progress} - progress proportion; if null progress bar not shown
   * @param {retry} - indicates if retry needed for error
   *
   * @returns {JSX.Element} the action status text, progress bar or blank view
   */
  const actionStatusTextAndVisual = useCallback(
    (flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress, retry = false) => {
      return (
        <>
          <View style={{ flex: flexSizeOuter, flexDirection: dimension, alignItems: 'center', justifyContent: 'center', width: '100%' }}>
            <View style={{ flex: flexSizeTextInner, justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}>
              <p id='Waiting for robot motion' className='waitingMsg' style={{ fontSize: waitingTextFontSize }}>
                {props.waitingText}
              </p>
              <p style={{ fontSize: motionTextFontSize }}>{text}</p>
              {showTime ? <p style={{ fontSize: motionTextFontSize }}>&nbsp;&nbsp;Elapsed Time: {time} sec</p> : <></>}
              {retry ? (
                <Button
                  variant='warning'
                  className='mx-2 btn-huge'
                  size='lg'
                  onClick={resumeCallback}
                  style={{
                    width: '90%',
                    height: '20%'
                  }}
                >
                  <h5 style={{ textAlign: 'center', fontSize: motionTextFontSize }}>Retry</h5>
                </Button>
              ) : (
                <></>
              )}
            </View>
            <View
              style={{
                flex: flexSizeVisualInner,
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              {progress === null ? <></> : <CircleProgressBar proportion={progress} />}
            </View>
          </View>
        </>
      )
    },
    [dimension, props.waitingText, motionTextFontSize, waitingTextFontSize, resumeCallback]
  )

  /**
   * Get the action status elements to render in a view flexbox.
   *
   * @returns {JSX.Element} the action status elements (e.g., robot motion text, elapsed time, progress bar)
   */
  const actionStatusText = useCallback(
    (actionStatus, flexSizeOuter) => {
      let flexSizeTextInner = 1
      let flexSizeVisualInner = 1
      let text = 'Robot is paused'
      let showTime = false
      let time = 0
      let progress = null
      switch (actionStatus.actionStatus) {
        case ROS_ACTION_STATUS_EXECUTE:
          if (actionStatus.feedback) {
            if (!actionStatus.feedback.is_planning) {
              let moving_elapsed_time = actionStatus.feedback.motion_time.sec + actionStatus.feedback.motion_time.nanosec / 10 ** 9
              text = 'Robot is moving...'
              time = Math.round(moving_elapsed_time * 100) / 100
              showTime = true
              progress = 1 - actionStatus.feedback.motion_curr_distance / actionStatus.feedback.motion_initial_distance
              // Calling CircleProgessBar component to visualize robot motion of moving
              return <>{actionStatusTextAndVisual(flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress)}</>
            } else {
              let planning_elapsed_time = actionStatus.feedback.planning_time.sec + actionStatus.feedback.planning_time.nanosec / 10 ** 9
              text = 'Robot is thinking...'
              time = Math.round(planning_elapsed_time * 100) / 100
              showTime = true
              return <>{actionStatusTextAndVisual(flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress)}</>
            }
          } else {
            // If you haven't gotten feedback yet, assume the robot is planning
            text = 'Robot is thinking...'
            return <>{actionStatusTextAndVisual(flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress)}</>
          }
        case ROS_ACTION_STATUS_SUCCEED:
          text = 'Robot has finished'
          return <>{actionStatusTextAndVisual(flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress)}</>
        case ROS_ACTION_STATUS_ABORT:
          /**
           * TODO: Just displaying that the robot faced an error is not useful
           * to the user. We should think more carefully about what different
           * error cases might arise, and change the UI accordingly to instruct
           * users on how to troubleshoot/fix it.
           */
          text = 'Robot encountered an error'
          return (
            <>{actionStatusTextAndVisual(flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress, true)}</>
          )
        case ROS_ACTION_STATUS_CANCELED:
          return <>{actionStatusTextAndVisual(flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress)}</>
        default:
          if (paused) {
            return <>{actionStatusTextAndVisual(flexSizeOuter, flexSizeTextInner, flexSizeVisualInner, text, showTime, time, progress)}</>
          } else {
            return (
              <View
                style={{ flex: flexSizeOuter, flexDirection: dimension, alignItems: 'center', justifyContent: 'center', width: '100%' }}
              >
                <h3>&nbsp;</h3>
              </View>
            )
          }
      }
    },
    [paused, dimension, actionStatusTextAndVisual]
  )

  // Render the component
  return (
    <>
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', width: '100%' }}>
        {props.debug ? (
          <Button variant='secondary' className='justify-content-center mx-2 mb-2' size='lg' onClick={robotMotionDone}>
            Continue (Debug Mode)
          </Button>
        ) : (
          <></>
        )}
        {actionStatusText(actionStatus, 1)}
      </View>
      {/**
       * Display the footer with the Pause button.
       */}
      <Footer
        pauseCallback={pauseCallback}
        backCallback={mealState === MEAL_STATE.R_MovingAbovePlate ? null : backCallback}
        backMealState={backMealState.current}
        resumeCallback={mealState === MEAL_STATE.R_BiteAcquisition ? null : resumeCallback}
        paused={paused}
      />
    </>
  )
}

RobotMotion.propTypes = {
  /**
   * Whether to run it in debug mode (e.g., if you aren't simulatenously running
   * the robot) or not
   */
  debug: PropTypes.bool.isRequired,
  // The meal state corresponding with the motion the robot is executing
  mealState: PropTypes.string.isRequired,
  // The meal state to transition to once the robot finishes executing
  nextMealState: PropTypes.string.isRequired,
  // The input to provide to the ROS action
  actionInput: PropTypes.object.isRequired,
  // The static text to display while the robot is executing the action
  waitingText: PropTypes.string.isRequired
}

export default RobotMotion
