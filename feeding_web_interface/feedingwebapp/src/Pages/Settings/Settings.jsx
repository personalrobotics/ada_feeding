// React imports
import React from 'react'
import Row from 'react-bootstrap/Row'
import Form from 'react-bootstrap/Form'

// Local imports
import { useGlobalState, SETTINGS } from '../GlobalState'
import ToggleButtonGroup from '../../buttons/ToggleButtonGroup'

/**
 * The Settings components displays all the settings users are able to configure.
 * Since settings are stored in global state, options for the settings should be
 * stored in GlobalState.js and this component should primarily focus on
 * rendering them.
 */
const Settings = () => {
  return (
    <div>
      {/**
       * The title of the page.
       */}
      <h1 style={{ textAlign: 'center', fontSize: '40px' }} className='txt-huge'>
        ⚙ Settings
      </h1>

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
      <Row className='justify-content-center mx-1 my-2'>
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
      </Row>
    </div>
  )
}

export default Settings
