/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React, { useCallback, useMemo, useState } from 'react'
import { View } from 'react-native'

// Local imports
import { useGlobalState, SETTINGS_STATE } from '../GlobalState'
import SettingsPageParent from './SettingsPageParent'

/**
 * The AutoContinue component allows users to change the auto-continue settings
 * from the Settings menu.
 */
const AutoContinue = () => {
  // Get relevant global state variables
  const setSettingsState = useGlobalState((state) => state.setSettingsState)
  const biteAcquisitionCheckAutoContinue = useGlobalState((state) => state.biteAcquisitionCheckAutoContinue)
  const setBiteAcquisitionCheckAutoContinue = useGlobalState((state) => state.setBiteAcquisitionCheckAutoContinue)
  const faceDetectionAutoContinue = useGlobalState((state) => state.faceDetectionAutoContinue)
  const setFaceDetectionAutoContinue = useGlobalState((state) => state.setFaceDetectionAutoContinue)
  const biteDoneAutoContinue = useGlobalState((state) => state.biteDoneAutoContinue)
  const setBiteDoneAutoContinue = useGlobalState((state) => state.setBiteDoneAutoContinue)

  // Rendering variables
  let textFontSize = '3.5vh'

  // Configure the parameters for SettingsPageParent
  const paramNames = useMemo(() => [], [])
  const [currentParams, setCurrentParams] = useState([])

  // Render the settings for the planning scene
  const renderAutoContinueSettings = useCallback(() => {
    return (
      <View
        style={{
          flex: 1,
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%',
          height: '100%'
        }}
      >
        <View
          style={{
            flex: 1,
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%'
          }}
        >
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            <input
              name='biteAcquisitionCheckAutoContinue'
              type='checkbox'
              checked={biteAcquisitionCheckAutoContinue}
              onChange={(e) => {
                setBiteAcquisitionCheckAutoContinue(e.target.checked)
              }}
              style={{ transform: 'scale(2.0)', verticalAlign: 'middle', marginRight: '15px' }}
            />
            Auto-continue after acquisition
          </p>
        </View>
        <View
          style={{
            flex: 1,
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%'
          }}
        >
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            <input
              name='faceDetectionAutoContinue'
              type='checkbox'
              checked={faceDetectionAutoContinue}
              onChange={(e) => {
                setFaceDetectionAutoContinue(e.target.checked)
              }}
              style={{ transform: 'scale(2.0)', verticalAlign: 'middle', marginRight: '15px' }}
            />
            Auto-continue after face detection
          </p>
        </View>
        <View
          style={{
            flex: 1,
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%'
          }}
        >
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize }}>
            <input
              name='biteDoneAutoContinue'
              type='checkbox'
              checked={biteDoneAutoContinue}
              onChange={(e) => {
                setBiteDoneAutoContinue(e.target.checked)
              }}
              style={{ transform: 'scale(2.0)', verticalAlign: 'middle', marginRight: '15px' }}
            />
            Auto-continue after transfer
          </p>
        </View>
      </View>
    )
  }, [
    textFontSize,
    biteAcquisitionCheckAutoContinue,
    setBiteAcquisitionCheckAutoContinue,
    faceDetectionAutoContinue,
    setFaceDetectionAutoContinue,
    biteDoneAutoContinue,
    setBiteDoneAutoContinue
  ])

  return (
    <SettingsPageParent
      title='Auto-Continue &#9881;'
      doneCallback={() => setSettingsState(SETTINGS_STATE.MAIN)}
      modalShow={false}
      modalOnHide={null}
      modalChildren={null}
      paramNames={paramNames}
      localParamValues={currentParams}
      setLocalParamValues={setCurrentParams}
    >
      {renderAutoContinueSettings()}
    </SettingsPageParent>
  )
}

export default AutoContinue
