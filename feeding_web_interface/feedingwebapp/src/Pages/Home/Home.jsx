/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React, { useCallback, useEffect, useMemo, useRef } from 'react'
import PropTypes from 'prop-types'
import { toast } from 'react-toastify'
import { View } from 'react-native'

// Local imports
import './Home.css'
import { createROSService, createROSServiceRequest, useROS } from '../../ros/ros_helpers'
import { useGlobalState, MEAL_STATE } from '../GlobalState'
import BiteAcquisitionCheck from './MealStates/BiteAcquisitionCheck'
import BiteDone from './MealStates/BiteDone'
import BiteSelection from './MealStates/BiteSelection'
import DetectingFace from './MealStates/DetectingFace'
import PostMeal from './MealStates/PostMeal'
import PreMeal from './MealStates/PreMeal'
import RobotMotion from './MealStates/RobotMotion'
import {
  ACQUISITION_REPORT_SERVICE_NAME,
  ACQUISITION_REPORT_SERVICE_TYPE,
  getRobotMotionText,
  REGULAR_CONTAINER_ID,
  TIME_TO_RESET_MS
} from '../Constants'

/**
 * The Home component displays the state of the meal, solicits user input as
 * needed, and communicates with the robot.
 *
 * @param {boolean} debug - whether to run it in debug mode (e.g., if you aren't
 *        simultaneously running the robot) or not
 * @returns {JSX.Element}
 */
function Home(props) {
  // Get the relevant values from global state
  const mealState = useGlobalState((state) => state.mealState)
  const mealStateTransitionTime = useGlobalState((state) => state.mealStateTransitionTime)
  const setBiteAcquisitionActionGoal = useGlobalState((state) => state.setBiteAcquisitionActionGoal)
  const setMoveToMouthActionGoal = useGlobalState((state) => state.setMoveToMouthActionGoal)
  const setMealState = useGlobalState((state) => state.setMealState)
  const setPaused = useGlobalState((state) => state.setPaused)
  const biteAcquisitionActionGoal = useGlobalState((state) => state.biteAcquisitionActionGoal)
  const moveToMouthActionGoal = useGlobalState((state) => state.moveToMouthActionGoal)
  const mostRecentBiteDoneResponse = useGlobalState((state) => state.mostRecentBiteDoneResponse)

  // Implement a wrapper around setMealState, that resets mostRecentBiteDoneResponse to
  // R_DetectingFace if the mealState is mostRecentBiteDoneResponse. In other words,
  // after you transition to that state once, you need to revisit BiteDone to transition
  // again.
  const setMealStateWrapper = useCallback(
    (newMealState) => {
      if (newMealState === mostRecentBiteDoneResponse) {
        setMealState(newMealState, MEAL_STATE.R_DetectingFace)
      } else {
        setMealState(newMealState)
      }
    },
    [mostRecentBiteDoneResponse, setMealState]
  )

  /**
   * Implement time-based transition of states. This is so that after the user
   * finishes a meal, when they start the next meal the app starts in PreMeal.
   * The `useEffect` with these parameters ensures that it is called when
   * reloading the page or when transitioning mealStates, but not when re-rendering.
   */
  useEffect(() => {
    if (Date.now() - mealStateTransitionTime >= TIME_TO_RESET_MS) {
      console.log('Reverting to PreMeal due to too much elapsed time in one state.')
      setBiteAcquisitionActionGoal(null)
      setMoveToMouthActionGoal(null)
      setMealState(MEAL_STATE.U_PreMeal)
      setPaused(false)
    }
  }, [mealStateTransitionTime, setMealState, setPaused, setMoveToMouthActionGoal, setBiteAcquisitionActionGoal])

  /**
   * All action inputs are constant. Note that we must be cautious if making
   * them non-constant, because the robot will re-execute an action every time
   * the action input changes (even on re-renders).
   */
  const moveAbovePlateActionInput = useMemo(() => ({}), [])
  const biteAcquisitionActionInput = useMemo(() => biteAcquisitionActionGoal, [biteAcquisitionActionGoal])
  const moveToRestingPositionActionInput = useMemo(() => ({}), [])
  const moveToStagingConfigurationActionInput = useMemo(() => ({}), [])
  const moveToMouthActionInput = useMemo(() => moveToMouthActionGoal, [moveToMouthActionGoal])
  const moveToStowPositionActionInput = useMemo(() => ({}), [])

  /**
   * Create callbacks for acquisition success and failure. This is done here because these
   * callbacks can be called during BiteAcquisition or the BiteAcquisitionCheck.
   */
  const lastMotionActionFeedback = useGlobalState((state) => state.lastMotionActionFeedback)
  const ros = useRef(useROS().ros)
  let acquisitionReportService = useRef(createROSService(ros.current, ACQUISITION_REPORT_SERVICE_NAME, ACQUISITION_REPORT_SERVICE_TYPE))
  let acquisitionResponse = useCallback(
    (success) => {
      if (!lastMotionActionFeedback.action_info_populated) {
        console.info('Cannot report acquisition success or failure without action_info_populated.')
        return
      }
      let msg, loss
      if (success) {
        msg = 'Reporting Food Acquisition Success!'
        loss = 0.0
      } else {
        msg = 'Reporting Food Acquisition Failure.'
        loss = 1.0
      }
      // NOTE: This uses the ToastContainer in Header
      console.log(msg)
      toast.info(msg, {
        containerId: REGULAR_CONTAINER_ID,
        toastId: msg
      })
      // Create a service request
      let request = createROSServiceRequest({
        loss: loss,
        action_index: lastMotionActionFeedback.action_index,
        posthoc: lastMotionActionFeedback.posthoc,
        id: lastMotionActionFeedback.selection_id
      })
      // Call the service
      let service = acquisitionReportService.current
      service.callService(request, (response) => console.log('Got acquisition report response', response))
    },
    [lastMotionActionFeedback]
  )

  /**
   * Determines what screen to render based on the meal state.
   */
  const getComponentByMealState = useCallback(() => {
    console.log('getComponentByMealState', mealState, props.debug)
    switch (mealState) {
      case MEAL_STATE.U_PreMeal: {
        return <PreMeal debug={props.debug} />
      }
      case MEAL_STATE.R_MovingAbovePlate: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingAbovePlate
        let nextMealState = MEAL_STATE.U_BiteSelection
        let backMealState = null
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            setMealState={setMealStateWrapper}
            nextMealState={nextMealState}
            backMealState={backMealState}
            actionInput={moveAbovePlateActionInput}
            waitingText={getRobotMotionText(currentMealState)}
          />
        )
      }
      case MEAL_STATE.U_BiteSelection: {
        return <BiteSelection debug={props.debug} webrtcURL={props.webrtcURL} />
      }
      case MEAL_STATE.R_BiteAcquisition: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_BiteAcquisition
        let nextMealState = MEAL_STATE.U_BiteAcquisitionCheck
        let backMealState = MEAL_STATE.R_MovingAbovePlate
        // TODO: Add an icon for this errorMealState!
        let errorMealState = MEAL_STATE.R_MovingToStagingConfiguration
        let errorCallback = () => acquisitionResponse(true) // Success if the user skips acquisition
        let errorMealStateDescription = 'Skip Acquisition'
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            setMealState={setMealStateWrapper}
            nextMealState={nextMealState}
            backMealState={backMealState}
            actionInput={biteAcquisitionActionInput}
            waitingText={getRobotMotionText(currentMealState)}
            allowRetry={false} // Don't allow retrying bite acquisition
            errorMealState={errorMealState}
            errorCallback={errorCallback}
            errorMealStateDescription={errorMealStateDescription}
          />
        )
      }
      case MEAL_STATE.R_MovingToRestingPosition: {
        let currentMealState = MEAL_STATE.R_MovingToRestingPosition
        let nextMealState = MEAL_STATE.U_BiteAcquisitionCheck
        let backMealState = MEAL_STATE.R_MovingAbovePlate
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            setMealState={setMealStateWrapper}
            nextMealState={nextMealState}
            backMealState={backMealState}
            actionInput={moveToRestingPositionActionInput}
            waitingText={getRobotMotionText(currentMealState)}
          />
        )
      }
      case MEAL_STATE.U_BiteAcquisitionCheck: {
        return <BiteAcquisitionCheck debug={props.debug} acquisitionResponse={acquisitionResponse} />
      }
      case MEAL_STATE.R_MovingToStagingConfiguration: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingToStagingConfiguration
        let nextMealState = MEAL_STATE.R_DetectingFace
        let backMealState = MEAL_STATE.R_MovingToRestingPosition
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            setMealState={setMealStateWrapper}
            nextMealState={nextMealState}
            backMealState={backMealState}
            actionInput={moveToStagingConfigurationActionInput}
            waitingText={getRobotMotionText(currentMealState)}
          />
        )
      }
      case MEAL_STATE.R_DetectingFace: {
        return <DetectingFace debug={props.debug} webrtcURL={props.webrtcURL} />
      }
      case MEAL_STATE.R_MovingToMouth: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingToMouth
        let nextMealState = MEAL_STATE.U_BiteDone
        let backMealState = MEAL_STATE.R_MovingFromMouth
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            setMealState={setMealStateWrapper}
            nextMealState={nextMealState}
            backMealState={backMealState}
            actionInput={moveToMouthActionInput}
            waitingText={getRobotMotionText(currentMealState)}
          />
        )
      }
      case MEAL_STATE.R_MovingFromMouth: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_MovingFromMouth
        let nextMealState = mostRecentBiteDoneResponse
        // Although slightly unintuitive, having backMealState being MovingAbovePlate
        // is necessary so the user doesn't get stuck in a situation where they can't
        // move above the plate.
        let backMealState = MEAL_STATE.R_MovingAbovePlate
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            setMealState={setMealStateWrapper}
            nextMealState={nextMealState}
            backMealState={backMealState}
            actionInput={moveToStagingConfigurationActionInput}
            waitingText={getRobotMotionText(currentMealState)}
          />
        )
      }
      case MEAL_STATE.U_BiteDone: {
        return <BiteDone debug={props.debug} />
      }
      case MEAL_STATE.R_StowingArm: {
        /**
         * We recreate currentMealState due to a race condition where sometimes
         * the app is performing a re-rendering and *then* the state is updated.
         */
        let currentMealState = MEAL_STATE.R_StowingArm
        let nextMealState = MEAL_STATE.U_PostMeal
        let backMealState = MEAL_STATE.R_MovingAbovePlate
        return (
          <RobotMotion
            debug={props.debug}
            mealState={currentMealState}
            setMealState={setMealStateWrapper}
            nextMealState={nextMealState}
            backMealState={backMealState}
            actionInput={moveToStowPositionActionInput}
            waitingText={getRobotMotionText(currentMealState)}
          />
        )
      }
      case MEAL_STATE.U_PostMeal: {
        return <PostMeal debug={props.debug} />
      }
      default: {
        return <div>Unknown meal state: {mealState}</div>
      }
    }
  }, [
    mealState,
    setMealStateWrapper,
    props.debug,
    props.webrtcURL,
    biteAcquisitionActionInput,
    acquisitionResponse,
    mostRecentBiteDoneResponse,
    moveAbovePlateActionInput,
    moveToMouthActionInput,
    moveToRestingPositionActionInput,
    moveToStowPositionActionInput,
    moveToStagingConfigurationActionInput
  ])

  // Render the component
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      {/**
       * The main contents of the screen depends on the mealState.
       */}
      {getComponentByMealState()}
    </View>
  )
}
Home.propTypes = {
  /**
   * Whether to run it in debug mode (e.g., if you aren't simultaneously running
   * the robot) or not
   */
  debug: PropTypes.bool,
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default Home
