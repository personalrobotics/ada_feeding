// React imports
import React, { useCallback, useEffect, useRef } from 'react'
import Button from 'react-bootstrap/Button'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'

/**
 * A button that responds to being held down. Specifically, as soon as the mouse
 * first goes down, this button will start an interval at a user-specified rate
 * that calls a user-specified function. When the mouse goes up or leaves the
 * button, the interval is cleared and the component calls a user-specified
 * cleanup function.
 *
 * @param {float} rate_hz - The rate at which to call the function, in Hz
 * @param {function} holdCallback - The function to call at the specified rate
 *    while the mouse is down. Note that this function must be static and cannot
 *    use any variables outputts by a React hook (e.g. useState, useRef, etc.)
 * @param {function} cleanupCallback - The function to call when the button stops
 *    being held. Note that this function must be static and cannot use any
 *    variables outputts by a React hook (e.g. useState, useRef, etc.)
 * @param {Object} buttonStyle - Optional style for the button
 *
 */
function HoldButton(props) {
  // Create a reference to the button
  const buttonRef = useRef(null)

  // Reference to the interval
  const intervalRef = useRef(null)

  // Callback to stop the interval
  const stopInterval = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
      let cleanupCallback = props.cleanupCallback
      cleanupCallback()
    }
  }, [props.cleanupCallback])

  // Callback to start the rate_hz interval
  const startInterval = useCallback(() => {
    // Stop the interval if it exists
    stopInterval()
    // Start a new interval
    intervalRef.current = setInterval(() => {
      let holdCallback = props.holdCallback
      holdCallback()
    }, 1000.0 / props.rate_hz)
  }, [
    props.rate_hz,
    props.holdCallback,
    stopInterval
    // setCounter
  ])

  // Callback for when the touch moves
  const onTouchMove = useCallback(
    (event) => {
      let { top, left, bottom, right } = buttonRef.current.getBoundingClientRect()
      if (
        event.touches === null ||
        event.touches.length < 1 ||
        event.touches[0].clientY < top ||
        event.touches[0].clientY > bottom ||
        event.touches[0].clientX < left ||
        event.touches[0].clientX > right
      ) {
        stopInterval()
      } else {
        startInterval()
      }
    },
    [buttonRef, startInterval, stopInterval]
  )

  // Stop the interval when the component is unmounted
  useEffect(() => {
    return stopInterval
  }, [stopInterval])

  return (
    <Button
      ref={buttonRef}
      variant={props.variant}
      style={{ ...props.buttonStyle, cursor: 'pointer' }}
      onMouseDown={startInterval}
      onMouseUp={stopInterval}
      onMouseLeave={stopInterval}
      onTouchStart={startInterval}
      onTouchEnd={stopInterval}
      onTouchCancel={stopInterval}
      onTouchMove={onTouchMove}
      onContextMenu={(e) => e.preventDefault()}
    >
      {props.children}
    </Button>
  )
}
HoldButton.propTypes = {
  rate_hz: PropTypes.number.isRequired,
  holdCallback: PropTypes.func.isRequired,
  cleanupCallback: PropTypes.func.isRequired,
  buttonStyle: PropTypes.object,
  children: PropTypes.node.isRequired,
  variant: PropTypes.string
}
HoldButton.defaultProps = {
  buttonStyle: {},
  variant: 'light'
}

export default HoldButton
