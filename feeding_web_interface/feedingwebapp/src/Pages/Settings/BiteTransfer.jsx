// React imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMediaQuery } from 'react-responsive'
import PropTypes from 'prop-types'
import { useId, Label, SpinButton } from '@fluentui/react-components'
import Button from 'react-bootstrap/Button'
import { View } from 'react-native'

// Local imports
import {
  getRobotMotionText,
  DISTANCE_TO_MOUTH_PARAM,
  MOVE_TO_MOUTH_SPEED_PARAM,
  MOVE_TO_MOUTH_SPEED_NEAR_MOUTH_PARAM,
  MOVE_FROM_MOUTH_SPEED_PARAM,
  MOVE_FROM_MOUTH_SPEED_NEAR_MOUTH_PARAM
} from '../Constants'
import { useGlobalState, MEAL_STATE, SETTINGS_STATE } from '../GlobalState'
import RobotMotion from '../Home/MealStates/RobotMotion'
import DetectingFaceSubcomponent from '../Home/MealStates/DetectingFaceSubcomponent'
import SettingsPageParent from './SettingsPageParent'

/**
 * The BiteTransfer component allows users to configure parameters related to the
 * bite transfer.
 */
const BiteTransfer = (props) => {
  // Get relevant global state variables
  const setSettingsState = useGlobalState((state) => state.setSettingsState)
  const globalMealState = useGlobalState((state) => state.mealState)
  const setPaused = useGlobalState((state) => state.setPaused)
  const settingsPageAtMouth = useGlobalState((state) => state.settingsPageAtMouth)
  const setSettingsPageAtMouth = useGlobalState((state) => state.setSettingsPageAtMouth)
  const moveToMouthActionGoal = useGlobalState((state) => state.moveToMouthActionGoal)

  // Create relevant local state variables
  // Configure the parameters for SettingsPageParent
  const paramNames = useMemo(
    () => [
      DISTANCE_TO_MOUTH_PARAM,
      MOVE_TO_MOUTH_SPEED_PARAM,
      MOVE_TO_MOUTH_SPEED_NEAR_MOUTH_PARAM,
      MOVE_FROM_MOUTH_SPEED_PARAM,
      MOVE_FROM_MOUTH_SPEED_NEAR_MOUTH_PARAM
    ],
    []
  )
  const [currentParams, setCurrentParams] = useState([null, null, null, null, null])
  const [localCurrAndNextMealState, setLocalCurrAndNextMealState] = useState(
    globalMealState === MEAL_STATE.U_BiteDone || globalMealState === MEAL_STATE.R_DetectingFace || settingsPageAtMouth
      ? [MEAL_STATE.R_MovingFromMouth, null]
      : [MEAL_STATE.R_MovingToStagingConfiguration, null]
  )
  const actionInput = useMemo(() => {
    if (localCurrAndNextMealState[0] === MEAL_STATE.R_MovingToMouth) {
      return moveToMouthActionGoal
    }
    return {}
  }, [localCurrAndNextMealState, moveToMouthActionGoal])
  const doneButtonIsClicked = useRef(false)

  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'
  // Rendering variables
  let textFontSize = '3.0vh'

  // Get min and max distance to mouth
  const minDistanceToMouth = 1 // cm
  const maxDistanceToMouth = 10 // cm
  const minLinearSpeed = 2 // cm/s
  const maxLinearSpeed = 15 // cm/s

  // When we set local meal state, also update bite transfer page at face
  const setLocalCurrMealStateWrapper = useCallback(
    (newLocalCurrMealState, newLocalNextMealState = null) => {
      let oldLocalCurrMealState = localCurrAndNextMealState[0]
      // If the oldlocalCurrMealState was R_MovingToMouth, then the robot is at the mouth
      setSettingsPageAtMouth(
        newLocalCurrMealState === null && (settingsPageAtMouth || oldLocalCurrMealState === MEAL_STATE.R_MovingToMouth)
      )
      // Start in a moving state, not a paused state
      setPaused(false)
      console.log('setLocalCurrMealStateWrapper', newLocalCurrMealState, newLocalNextMealState, doneButtonIsClicked.current)
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
      settingsPageAtMouth,
      localCurrAndNextMealState,
      setLocalCurrAndNextMealState,
      setSettingsPageAtMouth,
      doneButtonIsClicked,
      setPaused,
      setSettingsState
    ]
  )

  // Store the props for the RobotMotion call.
  const robotMotionProps = useMemo(() => {
    let localCurrMealState = localCurrAndNextMealState[0]
    let localNextMealState = localCurrAndNextMealState[1]
    return {
      mealState: localCurrMealState,
      setMealState: setLocalCurrMealStateWrapper,
      nextMealState: localNextMealState,
      backMealState: null,
      actionInput: actionInput,
      waitingText: localCurrMealState ? getRobotMotionText(localCurrMealState) : ''
    }
  }, [localCurrAndNextMealState, setLocalCurrMealStateWrapper, actionInput])

  // The first time the page is rendered, get the current distance to mouth
  useEffect(() => {
    doneButtonIsClicked.current = false
    // Since we start by moving to staging, this should be initialized to false
    setSettingsPageAtMouth(false)
    // Start in a moving state, not a paused state
    setPaused(false)
  }, [setSettingsPageAtMouth, setPaused, doneButtonIsClicked])

  // Callback to move the robot to the mouth
  const moveToMouthButtonClicked = useCallback(() => {
    setLocalCurrMealStateWrapper(MEAL_STATE.R_DetectingFace)
    doneButtonIsClicked.current = false
  }, [setLocalCurrMealStateWrapper, doneButtonIsClicked])

  // Callback to move the robot away from the mouth
  const moveAwayFromMouthButtonClicked = useCallback(() => {
    setLocalCurrMealStateWrapper(MEAL_STATE.R_MovingFromMouth)
    doneButtonIsClicked.current = false
  }, [setLocalCurrMealStateWrapper, doneButtonIsClicked])

  // Callback to return to the main settings page
  const doneButtonClicked = useCallback(() => {
    doneButtonIsClicked.current = true
    // Determine the state to move to based on the state before entering settings
    let localCurrMealState = MEAL_STATE.R_MovingFromMouth
    let localNextMealState
    // To get to Settings, the globalMealState must be one of the NON_MOVING_STATES
    switch (globalMealState) {
      case MEAL_STATE.U_BiteDone:
        if (settingsPageAtMouth) {
          localCurrMealState = null
          localNextMealState = null
        } else {
          localCurrMealState = MEAL_STATE.R_DetectingFace
        }
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
    setLocalCurrMealStateWrapper(localCurrMealState, localNextMealState)
  }, [settingsPageAtMouth, globalMealState, setLocalCurrMealStateWrapper, doneButtonIsClicked])

  // Callback for when the user changes the distance to mouth
  const onDistanceToMouthChange = useCallback(
    (_ev, data) => {
      let value = data.value !== null ? data.value : parseFloat(data.displayValue)
      if (value < minDistanceToMouth) {
        value = minDistanceToMouth
      }
      if (value > maxDistanceToMouth) {
        value = maxDistanceToMouth
      }
      let distance_m = value / 100.0
      setCurrentParams((currentParams) => [
        [distance_m, currentParams[0][1], currentParams[0][2]],
        currentParams[1],
        currentParams[2],
        currentParams[3],
        currentParams[4]
      ])
    },
    [minDistanceToMouth, maxDistanceToMouth]
  )

  // Callback for when the user changes the speed to the mouth
  const onSpeedChange = useCallback(
    (_ev, data, index) => {
      let value = data.value !== null ? data.value : parseFloat(data.displayValue)
      if (value < minLinearSpeed) {
        value = minLinearSpeed
      }
      if (value > maxLinearSpeed) {
        value = maxLinearSpeed
      }
      let speed_mps = value / 100.0
      setCurrentParams((currentParams) => [
        currentParams[0],
        index === 1 ? speed_mps : currentParams[1],
        index === 2 ? speed_mps : currentParams[2],
        index === 3 ? speed_mps : currentParams[3],
        index === 4 ? speed_mps : currentParams[4]
      ])
    },
    [minLinearSpeed, maxLinearSpeed]
  )

  // Callback to render the main contents of the page
  const distanceToMouthId = useId()
  const moveToMouthSpeedId = useId()
  const moveToMouthSpeedNearMouthId = useId()
  const moveFromMouthSpeedId = useId()
  const moveFromMouthSpeedNearMouthId = useId()
  const speedParameterIdsAndDescriptions = useMemo(
    () => [
      [moveToMouthSpeedId, 'Approach Speed (cm/s)'],
      [moveToMouthSpeedNearMouthId, 'Approach Near Mouth (cm/s)'],
      [moveFromMouthSpeedId, 'Retreat Speed (cm/s)'],
      [moveFromMouthSpeedNearMouthId, 'Retreat Near Mouth (cm/s)']
    ],
    [moveToMouthSpeedId, moveToMouthSpeedNearMouthId, moveFromMouthSpeedId, moveFromMouthSpeedNearMouthId]
  )
  const renderBiteTransferSettings = useCallback(() => {
    if (currentParams.some((param) => param === null)) {
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
          <View
            style={{
              flex: 16,
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
              value={currentParams[0][0] * 100}
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
            {speedParameterIdsAndDescriptions.map(([id, description], index) => (
              <>
                <Label
                  htmlFor={id}
                  style={{
                    fontSize: textFontSize,
                    width: '90%',
                    color: 'black',
                    textAlign: 'center'
                  }}
                >
                  {description}
                </Label>
                <SpinButton
                  value={currentParams[index + 1] * 100}
                  id={id}
                  step={1.0}
                  onChange={(_ev, data) => onSpeedChange(_ev, data, index + 1)}
                  appearance='filled-lighter'
                  style={{
                    fontSize: textFontSize,
                    width: '90%',
                    color: 'black'
                  }}
                  incrementButton={{
                    'aria-label': 'Increase value by 1.0',
                    'aria-roledescription': 'spinner',
                    size: 'large'
                  }}
                />
              </>
            ))}
          </View>
          <View
            style={{
              flex: 5,
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              zIndex: 1
            }}
          >
            <View
              style={{
                flex: 1,
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
            {/* <View
              style={{
                flex: 1,
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%'
              }}
            /> */}
            <View
              style={{
                flex: 1,
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
            {/* <View
              style={{
                flex: 1,
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%'
              }}
            /> */}
          </View>
        </View>
      )
    }
  }, [
    dimension,
    textFontSize,
    currentParams,
    onDistanceToMouthChange,
    onSpeedChange,
    distanceToMouthId,
    speedParameterIdsAndDescriptions,
    moveToMouthButtonClicked,
    moveAwayFromMouthButtonClicked
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
    <SettingsPageParent
      title='Bite Transfer &#9881;'
      doneCallback={doneButtonClicked}
      modalShow={localCurrAndNextMealState[0] !== null}
      modalOnHide={() => setLocalCurrMealStateWrapper(null)}
      modalChildren={renderModalBody()}
      paramNames={paramNames}
      localParamValues={currentParams}
      setLocalParamValues={setCurrentParams}
    >
      {renderBiteTransferSettings()}
    </SettingsPageParent>
  )
}
BiteTransfer.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default BiteTransfer
