// React imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMediaQuery } from 'react-responsive'
import Button from 'react-bootstrap/Button'
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local imports
import { useROS, createROSService, createROSServiceRequest } from '../../ros/ros_helpers'
import {
  CAMERA_FEED_TOPIC,
  getRobotMotionText,
  GET_JOINT_STATE_SERVICE_NAME,
  GET_JOINT_STATE_SERVICE_TYPE,
  ROBOT_JOINTS
} from '../Constants'
import { useGlobalState, MEAL_STATE, SETTINGS_STATE } from '../GlobalState'
import RobotMotion from '../Home/MealStates/RobotMotion'
import DetectingFaceSubcomponent from '../Home/MealStates/DetectingFaceSubcomponent'
import TeleopSubcomponent from '../Header/TeleopSubcomponent'
import SettingsPageParent from './SettingsPageParent'
import VideoFeed from '../Home/VideoFeed'

/**
 * The CustomizeConfiguration component allows users to configure the one of the
 * fixed configurations the robot uses. In its current form, the node can take in
 * multiple parameters, but will set them all to the same joint state positions.
 */
const CustomizeConfiguration = (props) => {
  // Check the props
  if (Object.values(MEAL_STATE).indexOf(props.startingMealState) === -1) {
    throw new Error('Invalid starting meal state ' + props.startingMealState)
  }

  // Get relevant global state variables
  const setSettingsState = useGlobalState((state) => state.setSettingsState)
  const globalMealState = useGlobalState((state) => state.mealState)
  const setPaused = useGlobalState((state) => state.setPaused)
  const biteTransferPageAtFace = useGlobalState((state) => state.biteTransferPageAtFace)

  // Create relevant local state variables
  // Configure the parameters for SettingsPageParent
  const [currentConfigurationParams, setCurrentConfigurationParams] = useState(props.paramNames.map(() => null))
  const [localCurrAndNextMealState, setLocalCurrAndNextMealState] = useState(
    globalMealState === MEAL_STATE.U_BiteDone || biteTransferPageAtFace
      ? [MEAL_STATE.R_MovingFromMouth, props.startingMealState]
      : [props.startingMealState, null]
  )
  const actionInput = useMemo(() => ({}), [])
  const doneButtonIsClicked = useRef(false)
  const [zoomLevel, setZoomLevel] = useState(1.0)
  const [mountTeleopSubcomponent, setMountTeleopSubcomponent] = useState(false)
  const unmountTeleopSubcomponentCallback = useRef(() => {})

  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'
  let otherDimension = isPortrait ? 'row' : 'column'
  // Rendering variables
  let textFontSize = 3.5
  let sizeSuffix = isPortrait ? 'vh' : 'vw'

  // Update other state variables that are related to the local meal state
  const setLocalCurrMealStateWrapper = useCallback(
    (newLocalCurrMealState, newLocalNextMealState = null) => {
      console.log('setLocalCurrMealStateWrapper evaluated')
      let oldLocalCurrMealState = localCurrAndNextMealState[0]
      // Only mount the teleop subcomponent if the robot finished the prereq motion for this page
      if (newLocalCurrMealState === null && oldLocalCurrMealState === props.startingMealState) {
        setMountTeleopSubcomponent(true)
      }
      // Start in a moving state, not a paused state
      setPaused(false)
      if (newLocalCurrMealState === null && doneButtonIsClicked.current) {
        // After the done button is clicked, the robot may have to do up to two
        // motions to restore itself to its old state. After that, this goes
        // back to the main settings page.
        setSettingsState(SETTINGS_STATE.MAIN)
      } else {
        setLocalCurrAndNextMealState([newLocalCurrMealState, newLocalNextMealState])
      }
    },
    [
      doneButtonIsClicked,
      localCurrAndNextMealState,
      props.startingMealState,
      setLocalCurrAndNextMealState,
      setMountTeleopSubcomponent,
      setPaused,
      setSettingsState
    ]
  )

  // Get the function that sets the local curr meal state and next meal state variables.
  // Note this does not execute it. It is used to pass the function to other components.
  const getSetLocalCurrMealStateWrapper = useCallback(
    (newLocalCurrMealState, newLocalNextMealState = null) => {
      let retval = () => {
        setLocalCurrMealStateWrapper(newLocalCurrMealState, newLocalNextMealState)
        unmountTeleopSubcomponentCallback.current = () => {}
      }
      // If the teleop subcomponent was mounted, unmount it and let the stopServo
      // success callback handle the rest. However, sometimes the success message
      // gets dropped over the network. So if the teleop subcomponent is already
      // unmounted, just call the retval function.
      if (mountTeleopSubcomponent) {
        setMountTeleopSubcomponent(false)
      } else {
        retval()
      }
      return retval
    },
    [mountTeleopSubcomponent, setLocalCurrMealStateWrapper, setMountTeleopSubcomponent, unmountTeleopSubcomponentCallback]
  )

  // Store the props for the RobotMotion call.
  const robotMotionProps = useMemo(() => {
    console.log('robotMotionProps set', actionInput)
    let localCurrMealState = localCurrAndNextMealState[0]
    let localNextMealState = localCurrAndNextMealState[1]
    return {
      mealState: localCurrMealState,
      // Robot motion should directly set the local curr meal state, since by
      // the time we are in the RobotMotion component, StopServo has already been
      // called.
      setMealState: setLocalCurrMealStateWrapper,
      nextMealState: localNextMealState,
      backMealState: null,
      actionInput: actionInput,
      waitingText: localCurrMealState ? getRobotMotionText(localCurrMealState) : ''
    }
  }, [localCurrAndNextMealState, setLocalCurrMealStateWrapper, actionInput])

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * Create the ROS Service Clients to get/set parameters.
   */
  let getJointStateService = useRef(createROSService(ros.current, GET_JOINT_STATE_SERVICE_NAME, GET_JOINT_STATE_SERVICE_TYPE))

  // Reset state the first time the page is rendered
  useEffect(() => {
    doneButtonIsClicked.current = false
    // Start in a moving state, not a paused state
    setPaused(false)
  }, [doneButtonIsClicked, setPaused])

  // Get the current joint states and store them as the above plate param
  const storeJointStatesAsLocalParam = useCallback(() => {
    console.log('storeJointStatesAsLocalParam called')
    let service = getJointStateService.current
    let request = createROSServiceRequest({
      joint_names: ROBOT_JOINTS
    })
    service.callService(request, (response) => {
      console.log('Got joint state response', response)
      setCurrentConfigurationParams(props.paramNames.map(() => response.joint_state.position))
    })
  }, [getJointStateService, props.paramNames, setCurrentConfigurationParams])

  // Callback to move the robot to another configuration
  const moveToButtonClicked = useCallback(
    (nextMealState) => {
      doneButtonIsClicked.current = false
      unmountTeleopSubcomponentCallback.current = getSetLocalCurrMealStateWrapper(nextMealState)
    },
    [getSetLocalCurrMealStateWrapper, doneButtonIsClicked, unmountTeleopSubcomponentCallback]
  )

  // Callback to return to the main settings page
  const doneButtonClicked = useCallback(() => {
    doneButtonIsClicked.current = true
    // Determine the state to move to based on the state before entering settings
    let localCurrMealState
    let localNextMealState = null
    // To get to Settings, the globalMealState must be one of the NON_MOVING_STATES
    switch (globalMealState) {
      case MEAL_STATE.U_BiteDone:
        localCurrMealState = MEAL_STATE.R_MovingToStagingConfiguration
        localNextMealState = MEAL_STATE.R_DetectingFace
        break
      case MEAL_STATE.U_PreMeal:
      case MEAL_STATE.U_BiteSelection:
        localCurrMealState = MEAL_STATE.R_MovingAbovePlate
        break
      case MEAL_STATE.U_BiteAcquisitionCheck:
        localCurrMealState = MEAL_STATE.R_MovingToRestingPosition
        break
      case MEAL_STATE.R_DetectingFace:
        localCurrMealState = MEAL_STATE.R_MovingToStagingConfiguration
        break
      case MEAL_STATE.U_PostMeal:
        localCurrMealState = MEAL_STATE.R_StowingArm
        break
      default:
        localCurrMealState = MEAL_STATE.R_MovingAbovePlate
        break
    }
    unmountTeleopSubcomponentCallback.current = getSetLocalCurrMealStateWrapper(localCurrMealState, localNextMealState)
  }, [getSetLocalCurrMealStateWrapper, globalMealState, doneButtonIsClicked, unmountTeleopSubcomponentCallback])

  // Callback to render the main contents of the page
  const renderSettings = useCallback(() => {
    return (
      <View
        style={{
          flex: 1,
          flexDirection: dimension,
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%',
          height: '100%'
        }}
      >
        <View style={{ flex: 8, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
          <VideoFeed topic={CAMERA_FEED_TOPIC} updateRateHz={10} webrtcURL={props.webrtcURL} zoom={zoomLevel} setZoom={setZoomLevel} />
        </View>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}></View>
        <View style={{ flex: 12, flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
          {mountTeleopSubcomponent ? (
            <>
              <View style={{ flex: 5, alignItems: 'center', justifyContent: 'center', width: '95%', height: '95%' }}>
                <TeleopSubcomponent
                  unmountCallback={unmountTeleopSubcomponentCallback}
                  teleopButtonOnReleaseCallback={storeJointStatesAsLocalParam}
                />
              </View>
            </>
          ) : (
            <p style={{ textAlign: 'center', fontSize: (textFontSize * 0.75).toString() + sizeSuffix, margin: 0 }} className='txt-huge'>
              To tune the &ldquo;{props.configurationName}&rdquo; configuration, first &ldquo;{props.buttonName}.&rdquo;
            </p>
          )}
        </View>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}></View>
        <View
          style={{
            flex: 3,
            flexDirection: otherDimension,
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          {props.otherButtonConfigs.map(({ name, mealState }) => (
            <View key={name} style={{ flex: 1, alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
              <Button
                variant='warning'
                className='mx-2 mb-2 btn-huge'
                size='lg'
                style={{
                  fontSize: (textFontSize * 0.5).toString() + sizeSuffix,
                  width: '90%',
                  height: '90%',
                  color: 'black'
                }}
                onClick={() => moveToButtonClicked(mealState)}
              >
                {name}
              </Button>
            </View>
          ))}
          <View style={{ flex: 1, flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
            <Button
              variant='warning'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              style={{
                fontSize: (textFontSize * 0.5).toString() + sizeSuffix,
                width: '90%',
                height: '90%',
                color: 'black'
              }}
              onClick={() => moveToButtonClicked(props.startingMealState)}
            >
              {props.buttonName}
            </Button>
          </View>
        </View>
      </View>
    )
  }, [
    dimension,
    otherDimension,
    textFontSize,
    sizeSuffix,
    zoomLevel,
    props.buttonName,
    props.configurationName,
    props.otherButtonConfigs,
    props.startingMealState,
    props.webrtcURL,
    mountTeleopSubcomponent,
    moveToButtonClicked,
    unmountTeleopSubcomponentCallback,
    storeJointStatesAsLocalParam
  ])

  // When a face is detected, switch to MoveToMouth
  const faceDetectedCallback = useCallback(() => {
    // This can directly call the setLocalCurrMealStateWrapper, since by the time
    // we get here, StopServo should have already been called.
    setLocalCurrMealStateWrapper(MEAL_STATE.R_MovingToMouth)
  }, [setLocalCurrMealStateWrapper])

  // Render the modal body, for calling robot code from within this settings page
  const renderModalBody = useCallback(() => {
    console.log('renderModalBody', localCurrAndNextMealState, robotMotionProps)
    let localCurrMealState = localCurrAndNextMealState[0]
    switch (localCurrMealState) {
      case MEAL_STATE.R_MovingToStagingConfiguration:
      case MEAL_STATE.R_MovingFromMouth:
      case MEAL_STATE.R_MovingToMouth:
      case MEAL_STATE.R_MovingAbovePlate:
      case MEAL_STATE.R_MovingToRestingPosition:
      case MEAL_STATE.R_StowingArm:
        return (
          <RobotMotion
            mealState={robotMotionProps.mealState}
            setMealState={robotMotionProps.setMealState}
            nextMealState={robotMotionProps.nextMealState}
            backMealState={robotMotionProps.backMealState}
            actionInput={robotMotionProps.actionInput}
            waitingText={robotMotionProps.waitingText}
          />
        )
      case MEAL_STATE.R_DetectingFace:
        return <DetectingFaceSubcomponent faceDetectedCallback={faceDetectedCallback} webrtcURL={props.webrtcURL} />
      default:
        return <></>
    }
  }, [localCurrAndNextMealState, props.webrtcURL, robotMotionProps, faceDetectedCallback])

  return (
    <SettingsPageParent
      title={props.configurationName + ' \u2699'}
      doneCallback={doneButtonClicked}
      modalShow={localCurrAndNextMealState[0] !== null}
      modalOnHide={() => setLocalCurrMealStateWrapper(null)}
      modalChildren={renderModalBody()}
      paramNames={props.paramNames}
      localParamValues={currentConfigurationParams}
      setLocalParamValues={setCurrentConfigurationParams}
      resetToPresetSuccessCallback={() => moveToButtonClicked(props.startingMealState)}
    >
      {renderSettings()}
    </SettingsPageParent>
  )
}
CustomizeConfiguration.propTypes = {
  // The meal state that must be executed before this page is rendered
  startingMealState: PropTypes.string.isRequired,
  // The names of the parameter this component should tune.
  paramNames: PropTypes.arrayOf(PropTypes.string).isRequired,
  // The name of the configuration this component tunes.
  configurationName: PropTypes.string.isRequired,
  // The name of the button that should be clicked to tune the configuration.
  buttonName: PropTypes.string.isRequired,
  // Other button configs
  otherButtonConfigs: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      mealState: PropTypes.string.isRequired
    })
  ).isRequired,
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default CustomizeConfiguration
