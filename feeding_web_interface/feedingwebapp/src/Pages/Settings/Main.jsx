// React imports
import React, { useCallback, useEffect, useRef, useState } from 'react'
import Button from 'react-bootstrap/Button'
import ButtonGroup from 'react-bootstrap/ButtonGroup'
import Dropdown from 'react-bootstrap/Dropdown'
import DropdownButton from 'react-bootstrap/DropdownButton'
import Modal from 'react-bootstrap/Modal'
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'
import Image from 'react-bootstrap/Image'
import { toast } from 'react-toastify'

// Local imports
import { useGlobalState, DEFAULT_NAMESPACE, MEAL_STATE, SETTINGS_STATE } from '../GlobalState'
import { useROS, createROSService, createROSServiceRequest, getValueFromParameter } from '../../ros/ros_helpers'
import {
  GET_PARAMETERS_SERVICE_NAME,
  GET_PARAMETERS_SERVICE_TYPE,
  MOVING_STATE_ICON_DICT,
  REGULAR_CONTAINER_ID,
  SET_PARAMETERS_SERVICE_NAME,
  SET_PARAMETERS_SERVICE_TYPE
} from '../Constants'

/**
 * The Main component displays all the settings users are able to configure.
 */
const Main = () => {
  // Create a local state variable to store whether the new preset modal is showing
  const [showNewPresetModal, setShowNewPresetModal] = useState(false)

  // Get relevant global state variables
  const setSettingsState = useGlobalState((state) => state.setSettingsState)
  const settingsPresets = useGlobalState((state) => state.settingsPresets)
  const setSettingsPresets = useGlobalState((state) => state.setSettingsPresets)

  // A reference for the new preset preset
  const newPresetInput = useRef(null)

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

  // Get the custom preset names from the robot
  const updatePresetsFromRobot = useCallback(() => {
    let service = getParametersService.current
    let request = createROSServiceRequest({
      names: ['custom_namespaces', 'namespace_to_use']
    })
    service.callService(request, (response) => {
      console.log('Got `custom_namespaces` response', response)
      if (response.values.length > 1 && response.values[0].type === 9 && response.values[1].type === 4) {
        setSettingsPresets({
          current: getValueFromParameter(response.values[1]),
          customNames: getValueFromParameter(response.values[0])
        })
      } else {
        setSettingsPresets({
          current: DEFAULT_NAMESPACE,
          customNames: []
        })
      }
    })
  }, [setSettingsPresets])

  // The first time this component is mounted, get the preset names
  useEffect(() => {
    updatePresetsFromRobot()
  }, [updatePresetsFromRobot])

  // Callback for when the user changes the preset
  const setPreset = useCallback(
    (preset, create_if_not_exist = true) => {
      console.log('setPreset', preset, create_if_not_exist)
      let presetOptions
      if (create_if_not_exist && preset !== DEFAULT_NAMESPACE && !settingsPresets.customNames.includes(preset)) {
        presetOptions = settingsPresets.customNames.concat([preset])
      } else {
        presetOptions = settingsPresets.customNames
      }

      let service = setParametersService.current
      let request = createROSServiceRequest({
        parameters: [
          {
            name: 'namespace_to_use',
            value: {
              type: 4, // string
              string_value: preset
            }
          },
          {
            name: 'custom_namespaces',
            value: {
              type: 9, // string array
              string_array_value: presetOptions
            }
          }
        ]
      })
      console.log('Calling service', request)
      service.callService(request, (response) => {
        console.log('Got response', response)
        if (response != null && response.results.length > 1 && response.results[0].successful && response.results[1].successful) {
          setSettingsPresets({
            current: preset,
            customNames: presetOptions
          })
        }
      })
    },
    [setParametersService, settingsPresets, setSettingsPresets]
  )

  // Callback for when users enter the new preset name
  const enterNewPresetName = useCallback(() => {
    // Get the new preset name
    let newPresetName = newPresetInput.current.value.trim()

    if (newPresetName.length === 0) {
      toast.error('Please enter a name for the new preset.', {
        containerId: REGULAR_CONTAINER_ID,
        toastId: 'newPresetName'
      })
      return
    } else {
      setPreset(newPresetName)
      setShowNewPresetModal(false)
    }
  }, [newPresetInput, setPreset])

  // Callback for when the user navigates to a settings page
  const onClickSettingsPage = useCallback(
    (settingsState) => {
      if (settingsPresets.current === DEFAULT_NAMESPACE) {
        toast.error('To change settings, select a preset other than `' + DEFAULT_NAMESPACE + '`.', {
          containerId: REGULAR_CONTAINER_ID,
          toastId: 'changePresets'
        })
      } else {
        setSettingsState(settingsState)
      }
    },
    [setSettingsState, settingsPresets]
  )

  // Get icon image for move to mouth
  let moveToMouthConfigurationImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth]
  let moveAbovePlateConfigurationImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingAbovePlate]

  // Configure the different options in the settings menu
  let settingsConfig = [
    {
      title: 'Bite Transfer',
      icon: moveToMouthConfigurationImage,
      onClick: () => onClickSettingsPage(SETTINGS_STATE.BITE_TRANSFER)
    },
    {
      title: 'Above Plate',
      icon: moveAbovePlateConfigurationImage,
      onClick: () => onClickSettingsPage(SETTINGS_STATE.ABOVE_PLATE)
    }
  ]

  return (
    <>
      <Container fluid>
        {/**
         * The title of the page.
         */}
        <Row className='justify-content-center mx-1 my-2'>
          <h1 style={{ textAlign: 'center', fontSize: '40px' }} className='txt-huge'>
            ⚙ Settings
          </h1>
        </Row>

        {/**
         * The preset settings element.
         */}
        <Row className='justify-content-center mx-1 my-2'>
          <Col>
            <p style={{ fontSize: '25px', textAlign: 'right', margin: '0rem' }}>Preset:</p>
          </Col>
          <Col>
            <DropdownButton
              as={ButtonGroup}
              key='settingsPresets'
              id={`dropdown-button-drop-down`}
              drop='down'
              variant='secondary'
              title={settingsPresets.current}
              size='lg'
            >
              <Dropdown.Item
                key={DEFAULT_NAMESPACE}
                onClick={() => setPreset(DEFAULT_NAMESPACE)}
                active={DEFAULT_NAMESPACE === settingsPresets.current}
              >
                {DEFAULT_NAMESPACE}
              </Dropdown.Item>
              <Dropdown.Divider />
              {settingsPresets.customNames.map((preset) => (
                <Dropdown.Item key={preset} onClick={() => setPreset(preset)} active={preset === settingsPresets.current}>
                  {preset}
                </Dropdown.Item>
              ))}
              <Dropdown.Divider />
              <Dropdown.Item onClick={() => setShowNewPresetModal(true)}>+ New Preset</Dropdown.Item>
            </DropdownButton>
          </Col>
        </Row>

        {/**
         * The button to navigate to the bite transfer settings page.
         */}
        {settingsConfig.map((config, i) => (
          <Row className='justify-content-center mx-1 my-2' key={i}>
            <Button
              variant='outline-dark'
              style={{
                fontSize: '30px',
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'center',
                alignItems: 'center',
                height: '8vh',
                borderWidth: '3px'
              }}
              onClick={config.onClick}
            >
              <Image
                fluid
                src={config.icon}
                style={{
                  height: '100%',
                  '--bs-btn-padding-x': '0rem',
                  '--bs-btn-padding-y': '0rem',
                  display: 'flex'
                }}
                alt={config.title}
                className='center'
              />
              <p
                style={{
                  display: 'flex',
                  marginTop: '1rem'
                }}
              >
                {config.title}
              </p>
            </Button>
          </Row>
        ))}
      </Container>

      {/**
       * A modal to allow users to specify the name of their new preset.
       */}
      <Modal
        show={showNewPresetModal}
        onHide={() => setShowNewPresetModal(false)}
        onShow={() => {
          if (newPresetInput.current !== null) {
            newPresetInput.current.focus()
          }
        }}
        size='lg'
        aria-labelledby='contained-modal-title-vcenter'
        backdrop='static'
        keyboard={false}
        centered
        id='newPresetModal'
      >
        <Modal.Header closeButton>
          <Modal.Title id='contained-modal-title-vcenter' style={{ fontSize: '30px' }}>
            Add New Preset
          </Modal.Title>
        </Modal.Header>
        <Modal.Body style={{ textAlign: 'center' }}>
          <input
            ref={newPresetInput}
            type='text'
            id='newPresetName'
            name='newPresetName'
            placeholder='Enter new preset name'
            style={{ fontSize: '25px', verticalAlign: 'middle' }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                enterNewPresetName()
              }
            }}
          />
          <br />
          <Button variant='secondary' style={{ fontSize: '25px', marginTop: '1rem' }} onClick={enterNewPresetName}>
            Create
          </Button>
        </Modal.Body>
      </Modal>
    </>
  )
}

export default Main
