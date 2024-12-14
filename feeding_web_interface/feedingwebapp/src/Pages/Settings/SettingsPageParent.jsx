/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React, { useCallback, useEffect, useRef } from 'react'
import { useMediaQuery } from 'react-responsive'
import Button from 'react-bootstrap/Button'
import Dropdown from 'react-bootstrap/Dropdown'
import SplitButton from 'react-bootstrap/SplitButton'
// The Modal is a screen that appears on top of the main app, and can be toggled
// on and off.
import Modal from 'react-bootstrap/Modal'
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local imports
import {
  useROS,
  createROSService,
  createROSServiceRequest,
  getValueFromParameter,
  getParameterFromValue,
  callROSAction,
  createROSActionClient,
  createROSMessage,
  destroyActionClient
} from '../../ros/ros_helpers'
import {
  GET_PARAMETERS_SERVICE_NAME,
  GET_PARAMETERS_SERVICE_TYPE,
  RECOMPUTE_WORKSPACE_WALLS_ACTION_NAME,
  RECOMPUTE_WORKSPACE_WALLS_ACTION_TYPE,
  SET_PARAMETERS_SERVICE_NAME,
  SET_PARAMETERS_SERVICE_TYPE
} from '../Constants'
import { DEFAULT_NAMESPACE, useGlobalState } from '../GlobalState'

/**
 * The SettingsPageParent component handles visual layout that is common across
 * all settings pages.
 */
const SettingsPageParent = (props) => {
  // Get relevant global state variables
  const settingsPresets = useGlobalState((state) => state.settingsPresets)

  // Get relevant local state variables
  const isResettingToPreset = useRef(false)
  const numTimesResetWorkspaceWalls = useRef(0)

  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Rendering variables
  let textFontSize = isPortrait ? 3.0 : 5.0
  let sizeSuffix = 'vh'

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

  /**
   * Create the ROS Action Client. This is created in useRef to avoid
   * re-creating it upon re-renders.
   */
  let recomputeWorkspaceWallsAction = useRef(
    createROSActionClient(ros.current, RECOMPUTE_WORKSPACE_WALLS_ACTION_NAME, RECOMPUTE_WORKSPACE_WALLS_ACTION_TYPE)
  )

  /**
   * Get the parameter values in the specified namespace or, if they don't exist,
   * in the default namespace. Set the local state to these values.
   */
  const setLocalParametersToGlobalValues = useCallback(
    (preset) => {
      if (props.paramNames.length === 0) {
        console.log('Skipping setLocalParametersToGlobalValues because there are no parameters to get.')
        return
      }

      let service = getParametersService.current
      let setLocalParamValues = props.setLocalParamValues

      // First, attempt to get the parameter values in the current namespace
      let currentRequest = createROSServiceRequest({
        names: props.paramNames.map((name) => preset.concat('.', name))
      })
      console.log('Sending GetParameter request for values in preset', preset, currentRequest)
      service.callService(currentRequest, (response) => {
        console.log('For request', currentRequest, 'received GetParameter response', response)
        let defaultRequest = createROSServiceRequest({
          names: []
        })
        let defaultRequestIndices = []
        let parameterValues = []
        for (let i = 0; i < response.values.length; i++) {
          if (response.values[i].type === 0 && preset !== DEFAULT_NAMESPACE) {
            // Parameter not set
            defaultRequest.names.push(DEFAULT_NAMESPACE.concat('.', props.paramNames[i]))
            defaultRequestIndices.push(i)
          } else {
            parameterValues.push(getValueFromParameter(response.values[i]))
          }
        }

        // For the parameters that are not set in the specified namespace, get the
        // default parameter values.
        if (defaultRequest.names.length > 0) {
          console.log('Sending GetParameter request for default values', defaultRequest)
          service.callService(defaultRequest, (response) => {
            console.log('For request', defaultRequest, 'received GetParameter response', response)
            for (let i = 0; i < response.values.length; i++) {
              parameterValues.splice(defaultRequestIndices[i], 0, getValueFromParameter(response.values[i]))
            }
            console.log('Setting local parameter values to', parameterValues)
            setLocalParamValues(parameterValues)
          })
        } else {
          console.log('Setting local parameter values to', parameterValues)
          setLocalParamValues(parameterValues)
        }
      })
    },
    [props.paramNames, props.setLocalParamValues, getParametersService]
  )

  /**
   * Get the values of the parameters the first time the component is rendered,
   * and set the local state to these values.
   */
  useEffect(() => {
    setLocalParametersToGlobalValues(settingsPresets.current)
  }, [setLocalParametersToGlobalValues, settingsPresets])

  /**
   * Save the current parameter values to the current namespace.
   */
  const setGlobalParameter = useCallback(() => {
    if (props.paramNames.length === 0) {
      console.log('Skipping setGlobalParameter because there are no parameters to set.')
      return
    }
    if (props.localParamValues.every((element) => element === null)) {
      console.log('Skipping setGlobalParameter because all values are null.')
      return
    }

    let service = setParametersService.current
    let currentRequest = createROSServiceRequest({
      parameters: props.paramNames.map((name, i) => {
        return {
          name: settingsPresets.current.concat('.', name),
          value: getParameterFromValue(props.localParamValues[i])
        }
      })
    })
    console.log('Sending SetParameter request', currentRequest)
    service.callService(currentRequest, (response) => {
      console.log('For request', currentRequest, 'received SetParameter response', response)
      // If it was successful, and if the props specify, then call the action to update the
      // workspace walls.
      if (response.result.successful) {
        if (props.resetWorkspaceWallsOnParameterUpdate) {
          console.log('Calling recompute_workspace_walls action')
          // numTimesResetWorkspaceWalls is used to ensure the result callback
          // only gets registered the first time. This is okay because none of the
          // values in the callback are expected to change.
          callROSAction(
            recomputeWorkspaceWallsAction.current,
            createROSMessage({}),
            null,
            numTimesResetWorkspaceWalls.current > 0
              ? null
              : (result) => {
                  console.log('Received result from recompute_workspace_walls action', result)
                  // If this is wrapping up a reset to preset, call the callback
                  if (isResettingToPreset.current) {
                    let resetToPresetSuccessCallback = props.resetToPresetSuccessCallback.current
                    resetToPresetSuccessCallback()
                    isResettingToPreset.current = false
                  }
                }
          )
          numTimesResetWorkspaceWalls.current++
        } else {
          // If this is wrapping up a reset to preset, call the callback
          if (isResettingToPreset.current) {
            let resetToPresetSuccessCallback = props.resetToPresetSuccessCallback.current
            resetToPresetSuccessCallback()
            isResettingToPreset.current = false
          }
        }
      }
    })
  }, [
    isResettingToPreset,
    numTimesResetWorkspaceWalls,
    props.localParamValues,
    props.paramNames,
    props.resetToPresetSuccessCallback,
    props.resetWorkspaceWallsOnParameterUpdate,
    setParametersService,
    settingsPresets
  ])

  /**
   * Every time the local parameter values change, update the global parameter.
   * The above frequency holds true because setGlobalParameter only changes when
   * props.localParamValues changes -- everything else should be constant.
   */
  useEffect(() => {
    setGlobalParameter()
  }, [setGlobalParameter])

  /**
   * When the component is unmounted, destroy the action client.
   */
  useEffect(() => {
    let action = recomputeWorkspaceWallsAction.current
    return () => {
      destroyActionClient(action)
    }
  }, [])

  /**
   * A callback for when the user asks to reset parameters to a preset.
   */
  const resetToPreset = useCallback(
    (preset) => {
      console.log('Resetting parameters to preset', preset)
      isResettingToPreset.current = true
      setLocalParametersToGlobalValues(preset)
    },
    [setLocalParametersToGlobalValues, isResettingToPreset]
  )

  return (
    <>
      <View
        style={{
          flex: 1,
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%'
        }}
      >
        {props.paramNames.length === 0 ? (
          <></>
        ) : (
          <View
            style={{
              flex: 4,
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'center',
              width: '100%',
              height: '100%',
              zIndex: 1
            }}
          >
            <p style={{ textAlign: 'center', fontSize: textFontSize.toString() + sizeSuffix, margin: 0 }} className='txt-huge'>
              {props.title}
            </p>
            <SplitButton
              variant='secondary'
              className='mx-2 mb-2 btn-huge'
              size='lg'
              style={{
                fontSize: textFontSize.toString() + sizeSuffix,
                marginLeft: '1rem'
              }}
              title={settingsPresets.current}
            >
              <Dropdown.Item key={DEFAULT_NAMESPACE} onClick={() => resetToPreset(DEFAULT_NAMESPACE)}>
                Reset parameter to {DEFAULT_NAMESPACE}
              </Dropdown.Item>
              {settingsPresets.customNames
                .filter((preset) => preset !== settingsPresets.current)
                .map((preset) => (
                  <Dropdown.Item key={preset} onClick={() => resetToPreset(preset)}>
                    Reset parameter to {preset}
                  </Dropdown.Item>
                ))}
            </SplitButton>
          </View>
        )}
        <View
          style={{
            flex: 32,
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          {props.children}
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
              fontSize: textFontSize.toString() + sizeSuffix,
              width: '90%',
              height: '90%',
              color: 'black',
              padding: '0rem'
            }}
            onClick={props.doneCallback}
          >
            Done
          </Button>
        </View>
      </View>
      <Modal
        show={props.modalShow}
        onHide={props.modalOnHide}
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
          <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', height: '65vh' }}>{props.modalChildren}</View>
        </Modal.Body>
      </Modal>
    </>
  )
}
SettingsPageParent.propTypes = {
  // The title of the page
  title: PropTypes.string.isRequired,
  // The content to include in the page
  children: PropTypes.node.isRequired,
  // Parameters for the modal
  modalShow: PropTypes.bool,
  modalOnHide: PropTypes.func,
  modalChildren: PropTypes.node,
  // Configure the parameters this component is in charge of.
  paramNames: PropTypes.arrayOf(PropTypes.string).isRequired,
  localParamValues: PropTypes.arrayOf(PropTypes.any).isRequired,
  setLocalParamValues: PropTypes.func.isRequired,
  // A ref that contains a function to call after the "reset to preset" button
  // successfully sets the local and global parameters. This must be a ref to avoid
  // unnecessary setting of global parameters.
  resetToPresetSuccessCallback: PropTypes.shape({
    current: PropTypes.func.isRequired
  }),
  // A prop to specify whether to invoke the action that resets workspace walls
  // when the global parameters have been updated
  resetWorkspaceWallsOnParameterUpdate: PropTypes.bool,
  // A function to call when the user is done with the page
  doneCallback: PropTypes.func.isRequired
}
SettingsPageParent.defaultProps = {
  modalShow: false,
  modalOnHide: () => {},
  modalChildren: <></>,
  resetToPresetSuccessCallback: {
    current: () => {}
  },
  resetWorkspaceWallsOnParameterUpdate: false
}

export default SettingsPageParent
