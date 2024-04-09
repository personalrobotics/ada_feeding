// React imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import PropTypes from 'prop-types'
import { useId, Label, SpinButton } from '@fluentui/react-components'
import Button from 'react-bootstrap/Button'
import Dropdown from 'react-bootstrap/Dropdown'
import SplitButton from 'react-bootstrap/SplitButton'
// The Modal is a screen that appears on top of the main app, and can be toggled
// on and off.
import Modal from 'react-bootstrap/Modal'
import { View } from 'react-native'

// Local imports
import { useROS, createROSService, createROSServiceRequest, getParameterValue } from '../../ros/ros_helpers'
import {
  GET_PARAMETERS_SERVICE_NAME,
  GET_PARAMETERS_SERVICE_TYPE,
  SET_PARAMETERS_SERVICE_NAME,
  SET_PARAMETERS_SERVICE_TYPE,
  DISTANCE_TO_MOUTH_PARAM
} from '../Constants'
import { useGlobalState, MEAL_STATE, SETTINGS_STATE, DEFAULT_NAMESPACE } from '../GlobalState'
import RobotMotion from '../Home/MealStates/RobotMotion'
import DetectingFaceSubcomponent from '../Home/MealStates/DetectingFaceSubcomponent'

/**
 * The BiteTransfer component allows users to configure parameters related to the
 * bite transfer.
 */
const BiteTransfer = (props) => {
  // Get relevant global state variables
  const settingsPresets = useGlobalState((state) => state.settingsPresets)
  const setSettingsState = useGlobalState((state) => state.setSettingsState)
  const globalMealState = useGlobalState((state) => state.mealState)
  const setPaused = useGlobalState((state) => state.setPaused)
  const biteTransferPageAtFace = useGlobalState((state) => state.biteTransferPageAtFace)
  const setBiteTransferPageAtFace = useGlobalState((state) => state.setBiteTransferPageAtFace)

  // Create relevant local state variables
  // Store the current distance to mouth
  const [currentDistanceToMouth, setCurrentDistanceToMouth] = useState(null)
  const [localCurrAndNextMealState, setLocalCurrAndNextMealState] = useState(
    globalMealState === MEAL_STATE.U_BiteDone || globalMealState === MEAL_STATE.R_DetectingFace || biteTransferPageAtFace
      ? [MEAL_STATE.R_MovingFromMouth, null]
      : [MEAL_STATE.R_MovingToStagingConfiguration, null]
  )
  const actionInput = useMemo(() => ({}), [])
  const [doneButtonIsClicked, setDoneButtonIsClicked] = useState(false)

  // Get min and max distance to mouth
  const minDistanceToMouth = 1 // cm
  const maxDistanceToMouth = 10 // cm

  // When we set local meal state, also update bite transfer page at face
  const setLocalCurrMealStateWrapper = useCallback(
    (newLocalCurrMealState, newLocalNextMealState = null) => {
      let oldLocalCurrMealState = localCurrAndNextMealState[0]
      // If the oldlocalCurrMealState was R_MovingToMouth, then the robot is at the mouth
      setBiteTransferPageAtFace(oldLocalCurrMealState === MEAL_STATE.R_MovingToMouth)
      // Start in a moving state, not a paused state
      setPaused(false)
      if (newLocalCurrMealState === null && doneButtonIsClicked) {
        // After the done button is clicked, the robot may have to do up to two
        // motions to restore itself to its old state. After that, this goes
        // back to the main settings page.
        setSettingsState(SETTINGS_STATE.MAIN)
      } else {
        setLocalCurrAndNextMealState([newLocalCurrMealState, newLocalNextMealState])
      }
    },
    [localCurrAndNextMealState, setLocalCurrAndNextMealState, setBiteTransferPageAtFace, doneButtonIsClicked, setPaused, setSettingsState]
  )

  // Store the props for the RobotMotion call. The first call has the robot move
  // to the staging configuration.
  const robotMotionProps = useMemo(() => {
    let localCurrMealState = localCurrAndNextMealState[0]
    let localNextMealState = localCurrAndNextMealState[1]
    let waitingText
    switch (localCurrMealState) {
      case MEAL_STATE.R_MovingToStagingConfiguration:
        waitingText = 'Waiting to move in front of you...'
        break
      case MEAL_STATE.R_MovingFromMouth:
        waitingText = 'Waiting to move away from you...'
        break
      case MEAL_STATE.R_MovingToMouth:
        waitingText = 'Waiting to move to your mouth...'
        break
      case MEAL_STATE.R_MovingAbovePlate:
        waitingText = 'Waiting to move above the plate...'
        break
      case MEAL_STATE.R_MovingToRestingPosition:
        waitingText = 'Waiting to move to the resting position...'
        break
      case MEAL_STATE.R_StowingArm:
        waitingText = 'Waiting to stow the arm...'
        break
      default:
        waitingText = 'Waiting to move in front of you...'
        break
    }
    console.log('useMemo called with', localCurrMealState, waitingText)
    return {
      mealState: localCurrMealState,
      setMealState: setLocalCurrMealStateWrapper,
      nextMealState: localNextMealState,
      backMealState: null,
      actionInput: actionInput,
      waitingText: waitingText
    }
  }, [localCurrAndNextMealState, setLocalCurrMealStateWrapper, actionInput])

  // Rendering variables
  let textFontSize = '3.5vh'

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * Create the ROS Service Clients to get/set parameters.
   */
  let getParametersService = useRef(createROSService(ros.current, GET_PARAMETERS_SERVICE_NAME, GET_PARAMETERS_SERVICE_TYPE))
  let setParametersService = useRef(createROSService(ros.current, SET_PARAMETERS_SERVICE_NAME, SET_PARAMETERS_SERVICE_TYPE))

  // The first time the page is rendered, get the current distance to mouth
  useEffect(() => {
    setDoneButtonIsClicked(false)
    // Start in a moving state, not a paused state
    setPaused(false)
    let service = getParametersService.current
    // First, attempt to get the current distance to mouth
    let currentRequest = createROSServiceRequest({
      names: [settingsPresets.current.concat('.', DISTANCE_TO_MOUTH_PARAM)]
    })
    service.callService(currentRequest, (response) => {
      console.log('Got current plan_distance_from_mouth response', response)
      if (response.values.length === 0 || response.values[0].type === 0) {
        // Parameter not set
        // Second, attempt to get the default distance to mouth
        let defaultRequest = createROSServiceRequest({
          names: [DEFAULT_NAMESPACE.concat('.', DISTANCE_TO_MOUTH_PARAM)]
        })
        service.callService(defaultRequest, (response) => {
          console.log('Got default plan_distance_from_mouth response', response)
          if (response.values.length > 0 && response.values[0].type === 8) {
            setCurrentDistanceToMouth(getParameterValue(response.values[0]))
          }
        })
      } else {
        setCurrentDistanceToMouth(getParameterValue(response.values[0]))
      }
    })
  }, [getParametersService, setCurrentDistanceToMouth, setDoneButtonIsClicked, setPaused, settingsPresets])

  // Callback to set the distance to mouth parameter
  const setDistanceToMouth = useCallback(
    (fullDistanceToMouth) => {
      let service = setParametersService.current
      let request = createROSServiceRequest({
        parameters: [
          {
            name: settingsPresets.current.concat('.', DISTANCE_TO_MOUTH_PARAM),
            value: {
              type: 8, // double array
              double_array_value: fullDistanceToMouth
            }
          }
        ]
      })
      service.callService(request, (response) => {
        console.log('Got response', response)
        if (response != null && response.results.length > 0 && response.results[0].successful) {
          setCurrentDistanceToMouth(fullDistanceToMouth)
        }
      })
    },
    [setParametersService, setCurrentDistanceToMouth, settingsPresets]
  )

  // Callback to restore the distance to mouth to the default
  const restoreToPreset = useCallback(
    (preset) => {
      console.log('restoreToPreset called with', preset)
      let service = getParametersService.current
      // Attempt to get the default distance to mouth
      let defaultRequest = createROSServiceRequest({
        names: [preset.concat('.', DISTANCE_TO_MOUTH_PARAM)]
      })
      service.callService(defaultRequest, (response) => {
        console.log('Got plan_distance_from_mouth response', response)
        if (response.values.length > 0 && response.values[0].type === 8) {
          setDistanceToMouth(getParameterValue(response.values[0]))
        } else {
          restoreToPreset(DEFAULT_NAMESPACE)
        }
      })
    },
    [getParametersService, setDistanceToMouth]
  )

  // Callback to move the robot to the mouth
  const moveToMouthButtonClicked = useCallback(() => {
    setLocalCurrMealStateWrapper(MEAL_STATE.R_DetectingFace)
    setDoneButtonIsClicked(false)
  }, [setLocalCurrMealStateWrapper, setDoneButtonIsClicked])

  // Callback to move the robot away from the mouth
  const moveAwayFromMouthButtonClicked = useCallback(() => {
    setLocalCurrMealStateWrapper(MEAL_STATE.R_MovingFromMouth)
    setDoneButtonIsClicked(false)
  }, [setLocalCurrMealStateWrapper, setDoneButtonIsClicked])

  // Callback to return to the main settings page
  const doneButtonClicked = useCallback(() => {
    setDoneButtonIsClicked(true)
    // Determine the state to move to based on the state before entering settings
    let localNextMealState
    // To get to Settings, the globalMealState must be one of the NON_MOVING_STATES
    switch (globalMealState) {
      case MEAL_STATE.U_BiteDone:
        localNextMealState = null
        break
      case MEAL_STATE.U_PreMeal:
      case MEAL_STATE.U_BiteSelection:
        localNextMealState = MEAL_STATE.R_MovingAbovePlate
        break
      case MEAL_STATE.U_BiteAcquisitionCheck:
        localNextMealState = MEAL_STATE.R_MovingToRestingPosition
        break
      case MEAL_STATE.U_PostMeal:
        localNextMealState = MEAL_STATE.R_StowingArm
        break
      default:
        localNextMealState = MEAL_STATE.R_MovingAbovePlate
        break
    }
    setLocalCurrMealStateWrapper(MEAL_STATE.R_MovingFromMouth, localNextMealState)
  }, [globalMealState, setLocalCurrMealStateWrapper, setDoneButtonIsClicked])

  // Callback for when the user changes the distance to mouth
  const onDistanceToMouthChange = useCallback(
    (_ev, data) => {
      let value = data.value ? data.value : parseFloat(data.displayValue)
      if (value < minDistanceToMouth) {
        value = minDistanceToMouth
      }
      if (value > maxDistanceToMouth) {
        value = maxDistanceToMouth
      }
      let fullDistanceToMouth = [value / 100.0, currentDistanceToMouth[1], currentDistanceToMouth[2]]
      setDistanceToMouth(fullDistanceToMouth)
    },
    [setDistanceToMouth, currentDistanceToMouth, minDistanceToMouth, maxDistanceToMouth]
  )

  // Callback to render the main contents of the page
  const distanceToMouthId = useId()
  const renderBiteTransferSettings = useCallback(() => {
    if (currentDistanceToMouth === null) {
      return (
        <>
          <View
            style={{
              flex: 1,
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%'
            }}
          >
            <h5 style={{ textAlign: 'center', fontSize: textFontSize }}>Loading...</h5>
          </View>
        </>
      )
    } else {
      return (
        <>
          <View
            style={{
              flex: 8,
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              zIndex: 1
            }}
          >
            <Label
              htmlFor={distanceToMouthId}
              style={{
                fontSize: textFontSize,
                width: '90%',
                color: 'black',
                textAlign: 'center'
              }}
            >
              Distance To Mouth (cm)
            </Label>
            <SpinButton
              value={currentDistanceToMouth[0] * 100}
              id={distanceToMouthId}
              step={0.5}
              onChange={onDistanceToMouthChange}
              appearance='filled-lighter'
              style={{
                fontSize: textFontSize,
                width: '90%',
                color: 'black'
              }}
              incrementButton={{
                'aria-label': 'Increase value by 0.5',
                'aria-roledescription': 'spinner',
                size: 'large'
              }}
            />
            <SplitButton
              variant='warning'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              style={{
                fontSize: textFontSize,
                width: '60%',
                color: 'black'
              }}
              title={'Set to '.concat(DEFAULT_NAMESPACE)}
              onClick={() => restoreToPreset(DEFAULT_NAMESPACE)}
            >
              {settingsPresets.customNames
                .filter((x) => x !== settingsPresets.current)
                .map((preset) => (
                  <Dropdown.Item key={preset} onClick={() => restoreToPreset(preset)}>
                    Set to {preset}
                  </Dropdown.Item>
                ))}
            </SplitButton>
          </View>
          <View
            style={{
              flex: 2,
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%'
            }}
          >
            <Button
              variant='warning'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              style={{
                fontSize: textFontSize,
                width: '90%',
                height: '90%',
                color: 'black'
              }}
              onClick={moveToMouthButtonClicked}
            >
              Move To Mouth
            </Button>
          </View>
          <View
            style={{
              flex: 2,
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%'
            }}
          />
          <View
            style={{
              flex: 2,
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%'
            }}
          >
            <Button
              variant='warning'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              style={{
                fontSize: textFontSize,
                width: '90%',
                height: '90%',
                color: 'black'
              }}
              onClick={moveAwayFromMouthButtonClicked}
            >
              Move From Mouth
            </Button>
          </View>
          <View
            style={{
              flex: 2,
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%'
            }}
          />
        </>
      )
    }
  }, [
    textFontSize,
    currentDistanceToMouth,
    onDistanceToMouthChange,
    distanceToMouthId,
    moveToMouthButtonClicked,
    moveAwayFromMouthButtonClicked,
    restoreToPreset,
    settingsPresets
  ])

  // When a face is detected, switch to MoveToMouth
  const faceDetectedCallback = useCallback(() => {
    setLocalCurrMealStateWrapper(MEAL_STATE.R_MovingToMouth)
  }, [setLocalCurrMealStateWrapper])

  // Render the modal body, for calling robot code from within this settings page
  const renderModalBody = useCallback(() => {
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
    <>
      <View
        style={{
          flex: 4,
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%'
        }}
      >
        <p style={{ textAlign: 'center', fontSize: textFontSize, margin: 0 }} className='txt-huge'>
          Bite Transfer Settings
        </p>
      </View>
      <View
        style={{
          flex: 3,
          flexDirection: 'row',
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%'
        }}
      >
        <View
          style={{
            flex: 1,
            justifyContent: 'center',
            alignItems: 'end',
            width: '100%'
          }}
        >
          <p style={{ fontSize: textFontSize, textAlign: 'right', margin: '0rem' }}>Preset:</p>
        </View>
        <View
          style={{
            flex: 1,
            justifyContent: 'center',
            alignItems: 'start',
            width: '100%'
          }}
        >
          <Button variant='secondary' disabled size='lg' style={{ marginLeft: '1rem' }}>
            {settingsPresets.current}
          </Button>
        </View>
      </View>
      <View
        style={{
          flex: 32,
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%'
        }}
      >
        {renderBiteTransferSettings()}
      </View>
      <View
        style={{
          flex: 4,
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%'
        }}
      >
        <Button
          variant='success'
          className='mx-2 mb-2 btn-huge'
          size='lg'
          style={{
            fontSize: textFontSize,
            width: '90%',
            height: '90%',
            color: 'black'
          }}
          onClick={doneButtonClicked}
        >
          Done
        </Button>
      </View>
      <Modal
        show={localCurrAndNextMealState[0] !== null}
        onHide={() => setLocalCurrMealStateWrapper(null)}
        size='lg'
        aria-labelledby='contained-modal-title-vcenter'
        backdrop='static'
        keyboard={false}
        centered
        id='robotMotionModal'
        fullscreen={false}
        dialogClassName='modal-90w'
        style={{
          '--bs-modal-padding': '0rem'
        }}
      >
        <Modal.Header closeButton />
        <Modal.Body style={{ overflow: 'hidden' }}>
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', height: '65vh' }}>{renderModalBody()}</View>
        </Modal.Body>
      </Modal>
    </>
  )
}
BiteTransfer.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default BiteTransfer
