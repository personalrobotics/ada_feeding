// React imports
import React from 'react'
import Button from 'react-bootstrap/Button'
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Image from 'react-bootstrap/Image'

// Local imports
import { MOVING_STATE_ICON_DICT } from '../Constants'
import { useGlobalState, /* SETTINGS, */ MEAL_STATE, SETTINGS_STATE } from '../GlobalState'
// import ToggleButtonGroup from '../../buttons/ToggleButtonGroup'

/**
 * The Main component displays all the settings users are able to configure.
 */
const Main = () => {
  // Get relevant global state variables
  const setSettingsState = useGlobalState((state) => state.setSettingsState)

  // Get icon image for move to mouth
  let moveToMouthConfigurationImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToMouth]

  return (
    <Container fluid>
      {/**
       * The title of the page.
       */}
      <Row className='justify-content-center mx-1 my-2'>
        <h1 style={{ textAlign: 'center', fontSize: '40px' }} className='txt-huge'>
          ⚙ Settings
        </h1>
      </Row>

      <Row className='justify-content-center mx-1 my-2'>
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
          onClick={() => setSettingsState(SETTINGS_STATE.BITE_TRANSFER)}
        >
          <Image
            fluid
            src={moveToMouthConfigurationImage}
            style={{
              height: '100%',
              '--bs-btn-padding-x': '0rem',
              '--bs-btn-padding-y': '0rem',
              display: 'flex'
            }}
            alt='move_to_mouth_image'
            className='center'
          />
          <p
            style={{
              display: 'flex',
              marginTop: '1rem'
            }}
          >
            Bite Transfer
          </p>
        </Button>
      </Row>

      {/**
       * Load toggle-able buttons for all the settings.
       *
       * TODO:
       *   - Instead of having a full sentence "title" for every setting, we
       *     should have a brief title with an optional "i" on the right side
       *     that users can click for additional information to pop up (perhaps
       *     as a Modal?)
       *   - We shouldn't assume all settings will be ToggleButtonGroup.
       *     For example, bite initiation settings should instead be checkboxes.
       */}
      {/* <Row className='justify-content-center mx-1 my-2'>
        <Form.Label style={{ fontSize: '30px' }}>Where should the robot wait after it gets food? </Form.Label>
        <ToggleButtonGroup
          valueOptions={SETTINGS.stagingPosition}
          currentValue={useGlobalState((state) => state.stagingPosition)}
          valueSetter={useGlobalState((state) => state.setStagingPosition)}
        />
      </Row>

      <Row className='justify-content-center mx-1 my-2'>
        <Form.Label style={{ fontSize: '30px' }}>How do you want to indicate readiness for a bite? </Form.Label>
        <ToggleButtonGroup
          valueOptions={SETTINGS.biteInitiation}
          currentValue={useGlobalState((state) => state.biteInitiation)}
          valueSetter={useGlobalState((state) => state.setBiteInitiation)}
        />
      </Row>

      <Row className='justify-content-center mx-1 my-2'>
        <Form.Label style={{ fontSize: '30px' }}>How do you want to select your desired food item? </Form.Label>
        <ToggleButtonGroup
          valueOptions={SETTINGS.biteSelection}
          currentValue={useGlobalState((state) => state.biteSelection)}
          valueSetter={useGlobalState((state) => state.setBiteSelection)}
        />
      </Row> */}
    </Container>
  )
}

export default Main
