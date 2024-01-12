// React Imports
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Button from 'react-bootstrap/Button'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'
// PropTypes is used to validate that the used props are in fact passed to this
// Component
import PropTypes from 'prop-types'

// Local Imports
import '../Home.css'
import { useROS, createROSActionClient, callROSAction, destroyActionClient } from '../../../ros/ros_helpers'
import { useWindowSize, convertRemToPixels } from '../../../helpers'
import MaskButton from '../../../buttons/MaskButton'
import {
  ROS_ACTIONS_NAMES,
  ROS_ACTION_STATUS_CANCEL_GOAL,
  ROS_ACTION_STATUS_EXECUTE,
  ROS_ACTION_STATUS_SUCCEED,
  ROS_ACTION_STATUS_ABORT,
  ROS_ACTION_STATUS_CANCELED,
  SEGMENTATION_STATUS_SUCCESS,
  MOVING_STATE_ICON_DICT
} from '../../Constants'
import { useGlobalState, MEAL_STATE } from '../../GlobalState'
import VideoFeed from '../VideoFeed'

/**
 * The BiteSelection component appears after the robot has moved above the plate,
 * plate. It enables users to select their desired food item.
 *
 * @param {boolean} debug - whether to run it in debug mode (e.g., if you aren't
 *        simulatenously running the robot) or not
 */
const BiteSelection = (props) => {
  // Get the relevant global variables
  const setMealState = useGlobalState((state) => state.setMealState)
  const setBiteAcquisitionActionGoal = useGlobalState((state) => state.setBiteAcquisitionActionGoal)
  // Get icon image for move to mouth
  let moveToStagingConfigurationImage = MOVING_STATE_ICON_DICT[MEAL_STATE.R_MovingToStagingConfiguration]

  // Reference to the DOM element of the parent of the video feed
  const videoParentRef = useRef(null)
  // Margin for the video feed and between the mask buttons. Note this cannot
  // be re-defined per render, otherwise it messes up re-rendering order upon
  // resize in VideoFeed.
  const margin = useMemo(() => convertRemToPixels(1), [])

  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Text font size
  let textFontSize = isPortrait ? '2.5vh' : '2vw'
  // Indicator of how to arrange screen elements based on orientation
  let dimension = isPortrait ? 'column' : 'row'
  // Whether to scale all the masks equally or not
  const scaleMasksEqually = false

  /**
   * Create a local state variable to store the detected masks, the
   * status of the action, and how many times the user has clicked the
   * image.
   */
  const [actionResult, setActionResult] = useState(null)
  const [numImageClicks, setNumImageClicks] = useState(0)
  /**
   * NOTE: We slightly abuse the ROS_ACTION_STATUS values in this local state
   * variable, by using it as a proxy for whether the robot is executing, has
   * succeeded, has been canceled, or has had an error. This is to avoid
   * creating an extra enum.
   */
  const [actionStatus, setActionStatus] = useState({
    actionStatus: null
  })

  /**
   * Connect to ROS, if not already connected. Put this in useRef to avoid
   * re-connecting upon re-renders.
   */
  const ros = useRef(useROS().ros)

  /**
   * Create the ROS Action Client. This is created in useRef to avoid
   * re-creating it upon re-renders.
   */
  let { actionName, messageType } = ROS_ACTIONS_NAMES[MEAL_STATE.U_BiteSelection]
  let segmentFromPointAction = useRef(createROSActionClient(ros.current, actionName, messageType))

  /**
   * Callback function for when the user indicates that they are done with their
   * meal.
   */
  const doneEatingClicked = useCallback(() => {
    console.log('doneEatingClicked')
    setMealState(MEAL_STATE.R_StowingArm)
  }, [setMealState])

  // Get current window size
  let windowSize = useWindowSize()

  /**
   * Callback function for when the user wants to move to mouth position.
   */
  const moveToStagingConfiguration = useCallback(() => {
    setMealState(MEAL_STATE.R_MovingToStagingConfiguration)
  }, [setMealState])

  /**
   * Callback function for when the user clicks the button for a food item.
   */
  const foodItemClicked = useCallback(
    (event) => {
      let food_i = Number(event.target.value)
      setBiteAcquisitionActionGoal({
        header: actionResult.header,
        camera_info: actionResult.camera_info,
        detected_food: actionResult.detected_items[food_i]
      })
      setMealState(MEAL_STATE.R_BiteAcquisition)
    },
    [actionResult, setBiteAcquisitionActionGoal, setMealState]
  )

  /**
   * Callback function for when the action sends feedback. It updates the
   * actionStatus local state variable.
   *
   * @param {object} feedbackMsg - the feedback message sent by the action
   */
  const feedbackCallback = useCallback(
    (feedbackMsg) => {
      setActionStatus({
        actionStatus: ROS_ACTION_STATUS_EXECUTE,
        feedback: feedbackMsg.values.feedback
      })
    },
    [setActionStatus]
  )

  /**
   * Callback function for when the action sends a response. It updates the
   * actionStatus local state variable and stores the returned masks if the
   * action succeeded.
   *
   * TODO: What should happen if the action succeeds, but a response is never
   * received e.g., due to a momentary networking issue? We should likely allow
   * the user to press a button to move on if a response hasn't been received
   * within n seconds of making the action call. Or allow the user to retry the
   * action call (maybe they would refresh the page anyway, which might be ok).
   */
  const responseCallback = useCallback(
    (response) => {
      if (response.response_type === 'result' && response.values.status === SEGMENTATION_STATUS_SUCCESS) {
        setActionStatus({
          actionStatus: ROS_ACTION_STATUS_SUCCEED
        })
        console.log('Got result', response.values)
        setActionResult(response.values)
      } else {
        if (
          response.response_type === 'cancel' ||
          response.values === ROS_ACTION_STATUS_CANCEL_GOAL ||
          response.values === ROS_ACTION_STATUS_CANCELED
        ) {
          setActionStatus({
            actionStatus: ROS_ACTION_STATUS_CANCELED
          })
        } else {
          setActionStatus({
            actionStatus: ROS_ACTION_STATUS_ABORT
          })
        }
      }
    },
    [setActionStatus, setActionResult]
  )

  /**
   * Callback function for when the user clicks the image of the plate.
   *
   * TODO: As we integrate this with SegmentAnything, we should think more
   * carefully about how to handle the case where the user clicks on a pixel,
   * changes their mind, and then clicks on another pixel. Ideally, we'd cancel
   * the old goal and only use the new goal. But first of all, that requires
   * canceling one (not all) active goals, which is not currently supported by
   * roslibjs ROS2 actions. We could implement this functionality on the action
   * server side, but that depends on whether calls to SegmentAnything are
   * blocking or not. If we do stick with the current functionality where users
   * clicks are disregarded until the previous goal is completed, then we should
   * make that clear to the user.
   */
  const imageClicked = useCallback(
    (x_raw, y_raw) => {
      // Call the food segmentation ROS action
      callROSAction(
        segmentFromPointAction.current,
        { seed_point: { header: { stamp: { sec: 0, nanosec: 0 }, frame_id: 'image_frame' }, point: { x: x_raw, y: y_raw, z: 0.0 } } },
        /**
         * Only register callbacks the first time to avoid multiple callbacks for
         * the same action call.
         */
        numImageClicks === 0 ? feedbackCallback : null,
        numImageClicks === 0 ? responseCallback : null
      )
      setNumImageClicks(numImageClicks + 1)
      // The current implementation of food segmentation does not send feedback,
      // so we manually set the action status to executing.
      setActionStatus({
        actionStatus: ROS_ACTION_STATUS_EXECUTE
      })
    },
    [segmentFromPointAction, numImageClicks, feedbackCallback, responseCallback, setNumImageClicks, setActionStatus]
  )

  /**
   * Cancel any running actions when the component unmounts
   */
  useEffect(() => {
    let action = segmentFromPointAction.current
    return () => {
      destroyActionClient(action)
    }
  }, [segmentFromPointAction])

  /** Get the continue button when debug mode is enabled
   *
   * @returns {JSX.Element} the continue debug button
   */
  const debugButton = useCallback(() => {
    // If the user is in debug mode, give them the option to skip
    return (
      <Button
        variant='secondary'
        className='justify-content-center mx-2 mb-2'
        size='lg'
        onClick={() => setMealState(MEAL_STATE.R_BiteAcquisition)}
        style={{ fontSize: textFontSize, width: '100%', height: '100%' }}
      >
        Continue (Debug Mode)
      </Button>
    )
  }, [setMealState, textFontSize])

  /**
   * Render the appropriate text and/or buttons based on the action status.
   *
   * @returns {string} the action status text to render
   */
  const actionStatusText = useCallback(() => {
    switch (actionStatus.actionStatus) {
      case ROS_ACTION_STATUS_EXECUTE:
        if (actionStatus.feedback) {
          let elapsed_time = actionStatus.feedback.elapsed_time.sec + actionStatus.feedback.elapsed_time.nanosec / 10 ** 9
          return 'Detecting food... ' + (Math.round(elapsed_time * 100) / 100).toString() + ' sec'
        } else {
          return 'Detecting food...'
        }
      case ROS_ACTION_STATUS_SUCCEED:
        if (actionResult && actionResult.detected_items && actionResult.detected_items.length > 0) {
          return 'Select a food, or retry image click.'
        } else {
          return 'Food detection succeeded'
        }
      case ROS_ACTION_STATUS_ABORT:
        /**
         * TODO: Just displaying that the robot faced an error is not useful
         * to the user. We should think more carefully about what different
         * error cases might arise, and change the UI accordingly to instruct
         * users on how to troubleshoot/fix it.
         */
        return 'Error in food detection'
      case ROS_ACTION_STATUS_CANCELED:
        return 'Food detection canceled'
      default:
        return ''
    }
  }, [actionStatus, actionResult])

  const maskButtonParentRef = useRef(null)
  /**
   * Renders the mask buttons
   *
   * @returns {JSX.Element} the mask buttons
   */
  const renderMaskButtons = useCallback(() => {
    // If the action succeeded
    if (actionStatus.actionStatus === ROS_ACTION_STATUS_SUCCEED) {
      // If we have a result and there are detected items
      if (actionResult && actionResult.detected_items && actionResult.detected_items.length > 0) {
        // Get the allotted space per mask
        let parentWidth, parentHeight
        if (maskButtonParentRef.current) {
          parentWidth = maskButtonParentRef.current.clientWidth
          parentHeight = maskButtonParentRef.current.clientHeight
        } else {
          /**
           * The below are initial guesses for the parent size based on our
           * allocation of views. As soon as the component is mounted, we will
           * use the actual parent size.
           */
          parentWidth = isPortrait ? windowSize.width : windowSize.width / 2.0
          parentHeight = isPortrait ? windowSize.height / 4.0 : windowSize.height / 3.0
        }
        let allottedSpaceWidth = parentWidth / actionResult.detected_items.length - margin * 2
        let allottedSpaceHeight = parentHeight - margin * 2
        let buttonSize = {
          width: allottedSpaceWidth,
          height: allottedSpaceHeight
        }

        /**
         * Determine how much to scale the masks so that the largest mask fits
         * into the alloted space.
         */
        // Get the size of the largest mask
        let [maxWidth, maxHeight] = [0, 0]
        if (scaleMasksEqually) {
          for (let detected_item of actionResult.detected_items) {
            if (detected_item.roi.width > maxWidth) {
              maxWidth = detected_item.roi.width
            }
            if (detected_item.roi.height > maxHeight) {
              maxHeight = detected_item.roi.height
            }
          }
        }
        // Create a list to contain the scale factors for each mask
        let maskScaleFactors = []
        for (let detected_item of actionResult.detected_items) {
          let widthScaleFactor = allottedSpaceWidth / (scaleMasksEqually ? maxWidth : detected_item.roi.width)
          let heightScaleFactor = allottedSpaceHeight / (scaleMasksEqually ? maxHeight : detected_item.roi.height)
          let maskScaleFactor = Math.min(widthScaleFactor, heightScaleFactor)
          maskScaleFactors.push(maskScaleFactor)
        }

        return (
          <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}>
            {actionResult.detected_items.map((detected_item, i) => (
              <View key={i} style={{ flex: 1, justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}>
                <MaskButton
                  imgSrc={'data:image/jpeg;base64,' + detected_item.rgb_image.data}
                  buttonSize={buttonSize}
                  maskSrc={'data:image/jpeg;base64,' + detected_item.mask.data}
                  invertMask={true}
                  maskScaleFactor={maskScaleFactors[i]}
                  maskBoundingBox={detected_item.roi}
                  onClick={foodItemClicked}
                  value={i.toString()}
                />
              </View>
            ))}
          </View>
        )
      }
    }
  }, [actionStatus, actionResult, foodItemClicked, isPortrait, windowSize, margin, scaleMasksEqually])

  /** Get the button for continue without acquiring bite
   *
   * @returns {JSX.Element} the skip acquisition button
   */
  const skipAcquisisitionButton = useCallback(() => {
    return (
      <>
        {/* Ask the user whether they want to skip acquisition and move above plate */}
        <View
          style={{
            flex: 1,
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          <h5 style={{ textAlign: 'center', fontSize: textFontSize }}>Skip acquisition</h5>
        </View>
        {/* Icon to move to mouth */}
        <View
          style={{
            flex: 4,
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          <Button
            variant='warning'
            className='mx-2 btn-huge'
            size='lg'
            onClick={moveToStagingConfiguration}
            style={{
              width: '90%',
              height: '90%'
            }}
          >
            <img
              src={moveToStagingConfigurationImage}
              style={{
                height: '90%',
                '--bs-btn-padding-x': '0rem',
                '--bs-btn-padding-y': '0rem'
              }}
              alt='move_to_mouth_image'
              className='center'
            />
          </Button>
        </View>
      </>
    )
  }, [moveToStagingConfiguration, moveToStagingConfigurationImage, textFontSize])

  /** Get the full page view
   *
   * @returns {JSX.Element} the the full page view
   */
  const fullPageView = useCallback(() => {
    return (
      <>
        {/**
         * In addition to selecting their desired food item, the user has two
         * other options on this page:
         *   - If their desired food item is not visible on the plate, they can
         *     decide to teleoperate the robot until it is visible.
         *   - Instead of selecting their next bite, the user can indicate that
         *     they are done eating.
         */}
        <View
          style={{
            flex: 3,
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            width: '100%'
          }}
        >
          <Button className='doneButton' style={{ fontSize: textFontSize, marginTop: '0', marginBottom: '0' }} onClick={doneEatingClicked}>
            ✅ Done Eating
          </Button>
        </View>
        {/**
         * Below the buttons, one half of the screen will present the video feed.
         * The other half will present the action status text, the food buttons
         * if the action has succeeded, and a button to proceed without acquiring
         * a bite.
         */}
        <View
          style={{
            flex: 17,
            flexDirection: dimension,
            alignItems: 'center',
            width: '100%'
          }}
        >
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
            <View
              style={{
                flex: 1,
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              <h5 style={{ textAlign: 'center', fontSize: textFontSize }}>Click on image to select food.</h5>
            </View>
            <View
              ref={videoParentRef}
              style={{
                flex: 9,
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              <VideoFeed parent={videoParentRef} pointClicked={imageClicked} webrtcURL={props.webrtcURL} />
            </View>
          </View>
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
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              <h5 style={{ textAlign: 'center', fontSize: textFontSize }}>{actionStatusText()}</h5>
            </View>
            <View
              style={{
                flex: 9,
                alignItems: 'center',
                justifyContent: 'center',
                width: '100%',
                height: '100%'
              }}
            >
              <View
                ref={maskButtonParentRef}
                style={{
                  flex: 4,
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '100%',
                  height: '100%'
                }}
              >
                {renderMaskButtons()}
              </View>
              <View
                style={{
                  flex: 4,
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '100%',
                  height: '100%'
                }}
              >
                {skipAcquisisitionButton()}
              </View>
              {props.debug ? (
                <View
                  style={{
                    flex: 1,
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '100%',
                    height: '100%'
                  }}
                >
                  {debugButton()}
                </View>
              ) : (
                <></>
              )}
            </View>
          </View>
        </View>
      </>
    )
  }, [
    doneEatingClicked,
    dimension,
    textFontSize,
    actionStatusText,
    renderMaskButtons,
    skipAcquisisitionButton,
    videoParentRef,
    imageClicked,
    props.debug,
    props.webrtcURL,
    debugButton
  ])

  // Render the component
  return <>{fullPageView()}</>
}
BiteSelection.propTypes = {
  /**
   * Whether to run it in debug mode (e.g., if you aren't simulatenously running
   * the robot) or not
   */
  debug: PropTypes.bool.isRequired,
  // The URL of the webrtc signalling server
  webrtcURL: PropTypes.string.isRequired
}

export default BiteSelection
