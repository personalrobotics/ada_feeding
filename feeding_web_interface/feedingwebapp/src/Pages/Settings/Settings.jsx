// React imports
import React, { useCallback } from 'react'
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local imports
import { useGlobalState, SETTINGS_STATE, MEAL_STATE } from '../GlobalState'
import Main from './Main'
import CustomizeConfiguration from './CustomizeConfiguration'
import BiteTransfer from './BiteTransfer'
import { ABOVE_PLATE_PARAM_JOINTS, RESTING_PARAM_JOINTS_1, RESTING_PARAM_JOINTS_2 } from '../Constants'

/**
 * The Settings components displays the appropriate settings page based on the
 * current settings state.
 */
const Settings = (props) => {
  // Get the relevant values from global state
  const settingsState = useGlobalState((state) => state.settingsState)

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
            paramNames={[ABOVE_PLATE_PARAM_JOINTS]}
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
            paramNames={[RESTING_PARAM_JOINTS_1, RESTING_PARAM_JOINTS_2]}
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
      default:
        console.log('Invalid settings state', settingsState)
        return <Main />
    }
  }, [props.webrtcURL, settingsState])

  // Render the component
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'start' }}>
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
