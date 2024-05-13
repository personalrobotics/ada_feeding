// React Imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Button from 'react-bootstrap/Button'
import { useId, Label, SpinButton } from '@fluentui/react-components'
import { useMediaQuery } from 'react-responsive'
import PropTypes from 'prop-types'
import { View } from 'react-native'

// Local Imports
import {
  useROS,
  createROSTopic,
  createROSMessage,
  createROSActionClient,
  callROSAction,
  destroyActionClient,
  createROSService,
  createROSServiceRequest,
  getParameterFromValue
} from '../../ros/ros_helpers'
import '../Home/Home.css'
import {
  CARTESIAN_CONTROLLER_NAME,
  JOINT_CONTROLLER_NAME,
  ROBOT_BASE_LINK,
  ROBOT_JOINTS,
  SERVO_CARTESIAN_TOPIC,
  SERVO_CARTESIAN_TOPIC_MSG,
  SERVO_JOINT_TOPIC,
  SERVO_JOINT_TOPIC_MSG,
  ACTIVATE_CONTROLLER_ACTION_NAME,
  ACTIVATE_CONTROLLER_ACTION_TYPE,
  SET_PARAMETERS_SERVICE_NAME,
  INCREASED_FORCE_THRESHOLD,
  DEFAULT_FORCE_THRESHOLD,
  FORCE_THRESHOLD_PARAM
} from '../Constants'
import { useGlobalState } from '../GlobalState'
import HoldButton from '../../buttons/HoldButton'

/**
 * The TeleopSubcomponent renders controls for the user to teleoperate the robot,
 * either in cartesian or joint space.
 */
const TeleopSubcomponent = (props) => {
  // Get local state
  const CARTESIAN_LINEAR_MODE = 'Move'
  const CARTESIAN_ANGULAR_MODE = 'Rotate'
  const JOINT_MODE = 'Joints'
  const [teleopMode, setTeleopMode] = useState(CARTESIAN_LINEAR_MODE)
  const [refreshCount, setRefreshCount] = useState(0)
  const speedSpinButtonID = useId()

  // Cartesian teleop speeds
  const LINEAR_MAX_SPEED = useMemo(() => 0.3, []) // m/s
  const LINEAR_MIN_SPEED = useMemo(() => 0.025, []) // m/s
  const teleopLinearSpeed = useGlobalState((state) => state.teleopLinearSpeed)
  const setTeleopLinearSpeed = useGlobalState((state) => state.setTeleopLinearSpeed)
  const ANGULAR_MAX_SPEED = useMemo(() => 0.6, []) // rad/s
  const ANGULAR_MIN_SPEED = useMemo(() => 0.05, []) // rad/s
  const teleopAngularSpeed = useGlobalState((state) => state.teleopAngularSpeed)
  const setTeleopAngularSpeed = useGlobalState((state) => state.setTeleopAngularSpeed)
  const JOINT_MAX_SPEED = useMemo(() => 0.6, []) // rad/s
  const JOINT_MIN_SPEED = useMemo(() => 0.05, []) // rad/s
  const teleopJointSpeed = useGlobalState((state) => state.teleopJointSpeed)
  const setTeleopJointSpeed = useGlobalState((state) => state.setTeleopJointSpeed)

  // Teleop publication frequency
  const rate_hz = useMemo(() => 10.0, [])

  // Style configuration
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  let textFontSize = isPortrait ? 2.2 : 2.5
  let sizeSuffix = isPortrait ? 'vh' : 'vw'
  const buttonStyle = useMemo(() => {
    return {
      width: '100%',
      height: '100%',
      border: '1px solid black',
      padding: '0rem',
      borderRadius: '0rem'
    }
  }, [])

  // Callback for when the user changes the speed
  const onSpeedChange = useCallback(
    (_ev, data) => {
      let value = data.value !== null ? data.value : parseFloat(data.displayValue)
      console.log('teleop value', value, data)
      let min_val =
        teleopMode === CARTESIAN_LINEAR_MODE
          ? LINEAR_MIN_SPEED
          : teleopMode === CARTESIAN_ANGULAR_MODE
          ? ANGULAR_MIN_SPEED
          : JOINT_MIN_SPEED
      let max_val =
        teleopMode === CARTESIAN_LINEAR_MODE
          ? LINEAR_MAX_SPEED
          : teleopMode === CARTESIAN_ANGULAR_MODE
          ? ANGULAR_MAX_SPEED
          : JOINT_MAX_SPEED
      if (value < min_val) {
        value = min_val
      }
      if (value > max_val) {
        value = max_val
      }
      let setter =
        teleopMode === CARTESIAN_LINEAR_MODE
          ? setTeleopLinearSpeed
          : teleopMode === CARTESIAN_ANGULAR_MODE
          ? setTeleopAngularSpeed
          : setTeleopJointSpeed
      setter(value)
    },
    [
      teleopMode,
      setTeleopLinearSpeed,
      setTeleopAngularSpeed,
      setTeleopJointSpeed,
      LINEAR_MIN_SPEED,
      LINEAR_MAX_SPEED,
      ANGULAR_MIN_SPEED,
      ANGULAR_MAX_SPEED,
      JOINT_MIN_SPEED,
      JOINT_MAX_SPEED
    ]
  )

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * Create the ROS Action Client to start the teleop controllers.
   */
  let activateControllerActionGoal = useMemo(
    () =>
      createROSMessage({
        controller_to_activate: teleopMode === JOINT_MODE ? JOINT_CONTROLLER_NAME : CARTESIAN_CONTROLLER_NAME,
        re_tare: true
      }),
    [teleopMode]
  )

  let activateControllerAction = useMemo(() => {
    return createROSActionClient(ros.current, ACTIVATE_CONTROLLER_ACTION_NAME, ACTIVATE_CONTROLLER_ACTION_TYPE)
  }, [])

  /**
   * Create the ROS Topic for cartesian and joint space teleoperation.
   */
  let cartesianTopic = useMemo(() => {
    return createROSTopic(ros.current, SERVO_CARTESIAN_TOPIC, SERVO_CARTESIAN_TOPIC_MSG)
  }, [])
  let jointTopic = useMemo(() => {
    return createROSTopic(ros.current, SERVO_JOINT_TOPIC, SERVO_JOINT_TOPIC_MSG)
  }, [])

  /**
   * When the component is mounted and when the refresh button is pressed, start servo.
   */
  useEffect(() => {
    console.log('Starting controller', refreshCount)
    let action = activateControllerAction
    callROSAction(action, activateControllerActionGoal, null, null)
  }, [refreshCount, activateControllerAction, activateControllerActionGoal])

  /**
   * When the component is unmounted, stop servo.
   */
  useEffect(() => {
    let unmountCallback = props.unmountCallback
    return () => {
      console.log('Unmounting teleop subcomponent.')
      destroyActionClient(activateControllerAction)
      unmountCallback.current()
    }
  }, [activateControllerAction, props.unmountCallback])

  /**
   * Callback function to publish constant cartesian cartesian velocity commands.
   */
  const publishCartesianVelocity = useCallback(
    (x, y, z, rx, ry, rz) => {
      if (x === 0.0 && y === 0.0 && z === 0.0 && rx === 0.0 && ry === 0.0 && rz === 0.0) {
        console.log('Publishing zero cartesian velocity.')
      } else {
        console.log('Publishing non-zero cartesian velocity.')
      }

      let msg = createROSMessage({
        header: {
          frame_id: ROBOT_BASE_LINK
        },
        twist: {
          linear: {
            x: x,
            y: y,
            z: z
          },
          angular: {
            x: rx,
            y: ry,
            z: rz
          }
        }
      })
      cartesianTopic.publish(msg)
    },
    [cartesianTopic]
  )

  /**
   * Callback function to publish a JointJog command to move a single joint.
   */
  const publishJointJog = useCallback(
    (joints, velocities) => {
      console.log('Publishing joint jog.')
      let msg = createROSMessage({
        header: {
          frame_id: ROBOT_BASE_LINK
        },
        joint_names: joints,
        velocities: velocities.map((velocity) => teleopJointSpeed * velocity),
        duration: 1.0 / rate_hz
      })
      jointTopic.publish(msg)
    },
    [jointTopic, rate_hz, teleopJointSpeed]
  )

  /**
   * Arrange the specified 6 elements in a trackpad-like arrangement.
   *
   * e11 and e12 are the "first-level," opposite from one another.
   * e21, and e22 are the "second-level," opposite from one another.
   * e31 and e32 are the "third-level," opposite from one another.
   */
  const trackpadArrangement = useCallback((e11, e12, e21, e22, e31, e32, horizontal = true) => {
    let mainFlexDirection = horizontal ? 'row' : 'column'
    let subFlexDirection = horizontal ? 'column' : 'row'

    return (
      <>
        <View
          style={{
            flex: 1,
            flexDirection: mainFlexDirection,
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          <View
            style={{
              flex: 1,
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              height: '100%'
            }}
          >
            {e11}
          </View>
          <View
            style={{
              flex: 4,
              flexDirection: subFlexDirection,
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              height: '100%'
            }}
          >
            <View
              style={{
                flex: 1,
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              {e21}
            </View>
            <View
              style={{
                flex: 2,
                flexDirection: mainFlexDirection,
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              <View
                style={{
                  flex: 2,
                  justifyContent: 'center',
                  alignItems: 'center',
                  width: '100%',
                  height: '100%'
                }}
              >
                {e31}
              </View>
              <View
                style={{
                  flex: 2,
                  justifyContent: 'center',
                  alignItems: 'center',
                  width: '100%',
                  height: '100%'
                }}
              >
                {e32}
              </View>
            </View>
            <View
              style={{
                flex: 1,
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              {e22}
            </View>
          </View>
          <View
            style={{
              flex: 1,
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              height: '100%'
            }}
          >
            {e12}
          </View>
        </View>
      </>
    )
  }, [])

  /**
   * Callback function to generate a button array of size major_axis*minor_axis and
   * fill it with specified elements row-wise.
   */
  const arrayArrangement = useCallback((major_axis, minor_axis, elements, horizontal = true) => {
    let majorFlexDirection = horizontal ? 'row' : 'column'
    let minorFlexDirection = horizontal ? 'column' : 'row'

    let layout = []
    for (let i = 0; i < major_axis; i++) {
      let row = []
      for (let j = 0; j < minor_axis; j++) {
        row.push(
          <View
            key={i.toString() + '-' + j.toString()}
            style={{
              flex: 1,
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              height: '100%'
            }}
          >
            {elements[i * minor_axis + j]}
          </View>
        )
      }
      layout.push(
        <View
          key={i.toString()}
          style={{
            flex: 1,
            flexDirection: minorFlexDirection,
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          {row}
        </View>
      )
    }
    return (
      <View
        style={{
          flex: 1,
          flexDirection: majorFlexDirection,
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%',
          height: '100%'
        }}
      >
        {layout}
      </View>
    )
  }, [])

  /**
   * Callback for when a teleop button gets released.
   */
  const stopTeleop = useCallback(
    (cartesian = true) => {
      let teleopButtonOnReleaseCallback = props.teleopButtonOnReleaseCallback
      if (cartesian) {
        publishCartesianVelocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      } else {
        publishJointJog(ROBOT_JOINTS, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      }
      teleopButtonOnReleaseCallback()
    },
    [publishCartesianVelocity, publishJointJog, props.teleopButtonOnReleaseCallback]
  )

  /**
   * Callback to get cartesian hold buttons
   */
  const getCartesianHoldButton = useCallback(
    (x, y, z, rx, ry, rz, text) => {
      return (
        <HoldButton
          rate_hz={rate_hz}
          holdCallback={() => {
            publishCartesianVelocity(
              teleopLinearSpeed * x,
              teleopLinearSpeed * y,
              teleopLinearSpeed * z,
              teleopAngularSpeed * rx,
              teleopAngularSpeed * ry,
              teleopAngularSpeed * rz
            )
          }}
          cleanupCallback={() => stopTeleop(true)}
          buttonStyle={buttonStyle}
        >
          {text}
        </HoldButton>
      )
    },
    [publishCartesianVelocity, buttonStyle, teleopLinearSpeed, teleopAngularSpeed, rate_hz, stopTeleop]
  )

  /**
   * Callback to get joint hold buttons
   */
  const getJointHoldButton = useCallback(
    (joint, velocity, text) => {
      return (
        <HoldButton
          rate_hz={rate_hz}
          holdCallback={() => {
            publishJointJog([joint], [velocity])
          }}
          cleanupCallback={() => stopTeleop(false)}
          buttonStyle={buttonStyle}
        >
          {text}
        </HoldButton>
      )
    },
    [publishJointJog, buttonStyle, rate_hz, stopTeleop]
  )

  /**
   * Callback to render the cartesian teleop controls.
   */
  const cartesianLinearTeleop = useCallback(() => {
    return trackpadArrangement(
      getCartesianHoldButton(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 'Up'),
      getCartesianHoldButton(0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 'Down'),
      getCartesianHoldButton(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Left'),
      getCartesianHoldButton(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Right'),
      getCartesianHoldButton(0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 'Away From You'),
      getCartesianHoldButton(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 'Towards You'),
      false
    )
  }, [trackpadArrangement, getCartesianHoldButton])

  /**
   * Callback to render the cartesian teleop controls.
   */
  const cartesianAngularTeleop = useCallback(() => {
    return trackpadArrangement(
      getCartesianHoldButton(0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 'Pitch +'),
      getCartesianHoldButton(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 'Pitch -'),
      getCartesianHoldButton(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 'Yaw +'),
      getCartesianHoldButton(0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 'Yaw -'),
      getCartesianHoldButton(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 'Roll +'),
      getCartesianHoldButton(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 'Roll -'),
      false
    )
  }, [trackpadArrangement, getCartesianHoldButton])

  /**
   * Callback to render the joint teleop controls.
   */
  const jointTeleop = useCallback(() => {
    // Swap J1, J2, J4 so that positive is "towards the user" and negative is "away"
    // for the above plate configuration
    let jointMultipliers = [-1, -1, 1, -4, 1, 1]
    let jointButtons = []
    for (let i = 0; i < ROBOT_JOINTS.length; i++) {
      jointButtons.push(
        getJointHoldButton(ROBOT_JOINTS[i], jointMultipliers[i] * 1.0, 'J' + (i + 1).toString() + '+'),
        getJointHoldButton(ROBOT_JOINTS[i], jointMultipliers[i] * -1.0, 'J' + (i + 1).toString() + '-')
      )
    }
    return arrayArrangement(ROBOT_JOINTS.length, 2, jointButtons, !isPortrait)
  }, [arrayArrangement, getJointHoldButton, isPortrait])

  /**
   * Callback to get a radio button corresponding to a teleop mode
   */
  const getTeleopRadioButton = useCallback(
    (mode) => {
      return (
        <View
          style={{
            flex: 1,
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          <p className='transitionMessage' style={{ marginBottom: '0px', fontSize: textFontSize.toString() + sizeSuffix }}>
            <input
              name='teleopMode'
              type='radio'
              checked={teleopMode === mode}
              onChange={(e) => {
                if (e.target.checked) {
                  setTeleopMode(mode)
                }
              }}
              style={{ transform: 'scale(1.5)', verticalAlign: 'middle', marginRight: '15px' }}
            />
            {mode}
          </p>
        </View>
      )
    },
    [teleopMode, setTeleopMode, textFontSize, sizeSuffix]
  )

  /**
   * Service and callback to increase the force threshold. Additionally, before
   * unmounting, the force threshold should be reset.
   */
  const [forceThresholdIsIncreased, setForceThresholdIsIncreased] = useState(false)
  let changeForceThresholdService = useMemo(() => {
    let activeController = teleopMode === JOINT_MODE ? JOINT_CONTROLLER_NAME : CARTESIAN_CONTROLLER_NAME
    return createROSService(ros.current, activeController + '/set_parameters_atomically', SET_PARAMETERS_SERVICE_NAME)
  }, [ros, teleopMode])
  const setForceThreshold = useCallback(
    (threshold) => {
      let service = changeForceThresholdService
      let request = createROSServiceRequest({
        parameters: [{ name: FORCE_THRESHOLD_PARAM, value: getParameterFromValue(threshold, 3) }]
      })
      console.log('Calling setForceThreshold with request', request, 'for service', service.name)
      service.callService(request, (response) => {
        console.log('For setForceThreshold request', request, 'received response', response)
        setForceThresholdIsIncreased(threshold > DEFAULT_FORCE_THRESHOLD)
      })
    },
    [changeForceThresholdService, setForceThresholdIsIncreased]
  )
  useEffect(() => {
    return () => {
      if (props.allowIncreasingForceThreshold) {
        setForceThreshold(DEFAULT_FORCE_THRESHOLD)
      }
    }
  }, [props.allowIncreasingForceThreshold, setForceThreshold])

  // Render the component
  return (
    <View
      style={{
        flex: 9,
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
          width: '100%',
          height: '100%'
        }}
      >
        {getTeleopRadioButton(CARTESIAN_LINEAR_MODE)}
        {getTeleopRadioButton(CARTESIAN_ANGULAR_MODE)}
        {getTeleopRadioButton(JOINT_MODE)}
      </View>
      <View
        style={{
          flex: 14,
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%',
          height: '100%'
        }}
      >
        {/* Allow users to tune to speed of the current teleop mode */}
        <View
          style={{
            flex: props.allowIncreasingForceThreshold ? 1 : 2,
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%'
          }}
        >
          <Button
            variant='secondary'
            className='mx-2 mb-2 btn-huge'
            size='lg'
            style={{
              fontSize: (textFontSize * 1.0).toString() + sizeSuffix,
              paddingTop: 0,
              paddingBottom: 0,
              margin: '0 !important'
            }}
            onClick={() => setRefreshCount(refreshCount + 1)}
          >
            &#10227;
          </Button>
          <Label
            htmlFor={speedSpinButtonID}
            style={{
              fontSize: textFontSize.toString() + sizeSuffix,
              width: '90%',
              color: 'black',
              textAlign: 'center'
            }}
          >
            Speed ({teleopMode === CARTESIAN_LINEAR_MODE ? 'm/s' : 'rad/s'})
          </Label>
          <SpinButton
            value={
              teleopMode === CARTESIAN_LINEAR_MODE
                ? teleopLinearSpeed
                : teleopMode === CARTESIAN_ANGULAR_MODE
                ? teleopAngularSpeed
                : teleopJointSpeed
            }
            id={speedSpinButtonID}
            step={0.025}
            onChange={onSpeedChange}
            appearance='filled-lighter'
            style={{
              fontSize: textFontSize.toString() + sizeSuffix,
              width: '90%',
              color: 'black'
            }}
            incrementButton={{
              'aria-label': 'Increase value by 0.5',
              'aria-roledescription': 'spinner',
              size: 'large'
            }}
          />
        </View>
        {/* If the props specify, show a button to increase the force threshold. */}
        {props.allowIncreasingForceThreshold ? (
          <View
            style={{
              flex: 1,
              flexDirection: 'row',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%'
            }}
          >
            {forceThresholdIsIncreased ? (
              <Button
                variant='warning'
                className='mx-2 mb-2 btn-huge'
                size='lg'
                style={{
                  fontSize: (textFontSize * 1.0).toString() + sizeSuffix,
                  paddingTop: 0,
                  paddingBottom: 0,
                  margin: '0 !important',
                  width: '100%'
                }}
                onClick={() => setForceThreshold(DEFAULT_FORCE_THRESHOLD)}
              >
                &#x26E8; Restore Force Threshold
              </Button>
            ) : (
              <Button
                variant='danger'
                className='mx-2 mb-2 btn-huge'
                size='lg'
                style={{
                  fontSize: (textFontSize * 1.0).toString() + sizeSuffix,
                  paddingTop: 0,
                  paddingBottom: 0,
                  margin: '0 !important',
                  width: '100%'
                }}
                onClick={() => setForceThreshold(INCREASED_FORCE_THRESHOLD)}
              >
                &#9888; Increase Force Threshold
              </Button>
            )}
          </View>
        ) : (
          <></>
        )}
        {/* Render the controls for the mode */}
        <View
          style={{
            flex: 8,
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%'
          }}
        >
          {teleopMode === CARTESIAN_LINEAR_MODE
            ? cartesianLinearTeleop()
            : teleopMode === CARTESIAN_ANGULAR_MODE
            ? cartesianAngularTeleop()
            : jointTeleop()}
        </View>
      </View>
    </View>
  )
}
TeleopSubcomponent.propTypes = {
  // A reference to a function to be called if StopServo is succesfully run.
  unmountCallback: PropTypes.object,
  // A function to be called when one of the teleop buttons are released
  teleopButtonOnReleaseCallback: PropTypes.func,
  // Whether to allow the user to increase the force threshold
  allowIncreasingForceThreshold: PropTypes.bool
}
TeleopSubcomponent.defaultProps = {
  unmountCallback: { current: () => {} },
  teleopButtonOnReleaseCallback: () => {},
  allowIncreasingForceThreshold: false
}
export default TeleopSubcomponent
