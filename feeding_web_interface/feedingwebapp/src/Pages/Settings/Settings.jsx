// React imports
import React, { useCallback, useMemo } from 'react'
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local imports
import { useGlobalState, SETTINGS_STATE, MEAL_STATE } from '../GlobalState'
import Main from './Main'
import CustomizeConfiguration, {
  getJointPositionsFromRobotStateResponse,
  getEndEffectorPositionFromRobotStateResponse,
  getEndEffectorOrientationFromRobotStateResponse
} from './CustomizeConfiguration'
import BiteTransfer from './BiteTransfer'
import {
  ABOVE_PLATE_PARAM_JOINTS,
  FACE_DETECTION_IMG_TOPIC,
  RESTING_PARAM_JOINTS_1,
  RESTING_PARAM_JOINTS_2,
  STAGING_PARAM_JOINTS,
  STAGING_PARAM_ORIENTATION,
  STAGING_PARAM_POSITION,
  STOW_PARAM_JOINTS
} from '../Constants'
import PlanningScene from './PlanningScene'

/**
 * The Settings components displays the appropriate settings page based on the
 * current settings state.
 */
const Settings = (props) => {
  // Get the relevant values from global state
  const settingsState = useGlobalState((state) => state.settingsState)

  // Get the param names, to avoid re-creating them on re-renders
  const abovePlateParamNames = useMemo(() => [ABOVE_PLATE_PARAM_JOINTS], [])
  const restingParamNames = useMemo(() => [RESTING_PARAM_JOINTS_1, RESTING_PARAM_JOINTS_2], [])
  const stagingParamNames = useMemo(() => [STAGING_PARAM_JOINTS, STAGING_PARAM_POSITION, STAGING_PARAM_ORIENTATION], [])
  const stowParamNames = useMemo(() => [STOW_PARAM_JOINTS], [])

  const getComponentBySettingsState = useCallback(() => {
    console.log('getComponentBySettingsState', settingsState)
    switch (settingsState) {
      case SETTINGS_STATE.MAIN:
        return <Main />
      case SETTINGS_STATE.BITE_TRANSFER:
        return <BiteTransfer webrtcURL={props.webrtcURL} />
      case SETTINGS_STATE.ABOVE_PLATE:
        return (
          <CustomizeConfiguration
            startingMealState={MEAL_STATE.R_MovingAbovePlate}
            paramNames={abovePlateParamNames}
            getEndEffectorPose={false}
            getParamValues={[getJointPositionsFromRobotStateResponse]}
            configurationName='Above Plate'
            buttonName='Move Above Plate'
            otherButtonConfigs={[
              {
                name: 'Move to Staging',
                mealState: MEAL_STATE.R_MovingToStagingConfiguration
              },
              {
                name: 'Move to Resting',
                mealState: MEAL_STATE.R_MovingToRestingPosition
              }
            ]}
            webrtcURL={props.webrtcURL}
          />
        )
      case SETTINGS_STATE.RESTING_CONFIGURATION:
        return (
          <CustomizeConfiguration
            startingMealState={MEAL_STATE.R_MovingToRestingPosition}
            paramNames={restingParamNames}
            getEndEffectorPose={false}
            getParamValues={[getJointPositionsFromRobotStateResponse, getJointPositionsFromRobotStateResponse]}
            configurationName='Resting Position'
            buttonName='Move to Resting'
            otherButtonConfigs={[
              {
                name: 'Move to Staging',
                mealState: MEAL_STATE.R_MovingToStagingConfiguration
              },
              {
                name: 'Move Above Plate',
                mealState: MEAL_STATE.R_MovingAbovePlate
              }
            ]}
            webrtcURL={props.webrtcURL}
          />
        )
      case SETTINGS_STATE.STAGING_CONFIGURATION:
        return (
          <CustomizeConfiguration
            startingMealState={MEAL_STATE.R_MovingToStagingConfiguration}
            paramNames={stagingParamNames}
            getEndEffectorPose={true}
            getParamValues={[
              getJointPositionsFromRobotStateResponse,
              getEndEffectorPositionFromRobotStateResponse,
              getEndEffectorOrientationFromRobotStateResponse
            ]}
            configurationName='Staging Position'
            buttonName='Move to Staging'
            otherButtonConfigs={[
              {
                name: 'Move to Mouth',
                mealState: MEAL_STATE.R_DetectingFace
              },
              {
                name: 'Move Above Plate',
                mealState: MEAL_STATE.R_MovingAbovePlate
              },
              {
                name: 'Move to Resting',
                mealState: MEAL_STATE.R_MovingToRestingPosition
              }
            ]}
            toggleFaceDetection={true}
            videoTopic={FACE_DETECTION_IMG_TOPIC}
            webrtcURL={props.webrtcURL}
          />
        )
      case SETTINGS_STATE.STOW_CONFIGURATION:
        return (
          <CustomizeConfiguration
            startingMealState={MEAL_STATE.R_StowingArm}
            paramNames={stowParamNames}
            getEndEffectorPose={false}
            getParamValues={[getJointPositionsFromRobotStateResponse]}
            configurationName='Stow Position'
            buttonName='Stow Arm'
            otherButtonConfigs={[
              {
                name: 'Move Above Plate',
                mealState: MEAL_STATE.R_MovingAbovePlate
              }
            ]}
            webrtcURL={props.webrtcURL}
          />
        )
      case SETTINGS_STATE.PLANNING_SCENE:
        return <PlanningScene />
      default:
        console.log('Invalid settings state', settingsState)
        return <Main />
    }
  }, [abovePlateParamNames, restingParamNames, stagingParamNames, stowParamNames, props.webrtcURL, settingsState])

  // Render the component
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'start', width: '100%', height: '100%' }}>
      {/**
       * The main contents of the screen depends on the settingsState.
       */}
      {getComponentBySettingsState()}
    </View>
  )
}
Settings.propTypes = {
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default Settings
