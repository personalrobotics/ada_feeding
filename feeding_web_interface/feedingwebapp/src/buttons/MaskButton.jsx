/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React, { useCallback, useEffect, useRef } from 'react'
import Button from 'react-bootstrap/Button'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'
import { useWindowSize } from '../helpers'

/**
 * A component that renders an image and a mask on top of it, all within a
 * button.
 *
 * @param {Object} buttonSize - The size of the button
 * @param {string} imgSrc - The source of the image to be displayed
 * @param {Object} imgSize - The size of the image
 * @param {string} maskSrc - The source of the mask to be displayed
 * @param {boolean} invertMask - Whether or not to invert the mask. In either
 *     case, what is white in the original mask will show and what is black will
 *     be hidden. What differs is whether the outside of the mask is black (false)
 *     or white (true).
 * @param {number} maskScaleFactor - The factor by which to scale the mask. This
 *     is necessary because the mask is in the size of the original image from
 *     the robot's camera, but we scale that image down to fit the screen. The
 *     scale factor is essentially the ratio of imgSize to the original image
 *     size.
 * @param {Object} maskBoundingBox - The bounding box of the mask. In other words,
 *     where is the mask located relative to the original image's origin?
 * @param {function} onClick - The function to call when the button is clicked
 * @param {number} value - The value to pass to the onClick function
 *
 */
function MaskButton(props) {
  // Get a reference for the canvas
  const canvasRef = useRef(null)

  // Get the properties
  let buttonSize = props.buttonSize
  let imgSrc = props.imgSrc
  let maskSrc = props.maskSrc
  let invertMask = props.invertMask
  let maskScaleFactor = props.maskScaleFactor
  let maskBoundingBox = props.maskBoundingBox
  let onClick = props.onClick
  let value = props.value

  // Draw a red filled circle on the middle of the canvas
  const drawCircle = useCallback(() => {
    // Get the canvas
    const canvas = canvasRef.current
    // Canvas might be null if the previous render had more MaskButtons
    // than the current render.
    if (canvas !== null) {
      // Get the context
      const ctx = canvas.getContext('2d')
      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      // Draw a red filled circle
      let radius = 5
      ctx.beginPath()
      ctx.arc(canvas.width / 2, canvas.height / 2, radius, 0, 2 * Math.PI)
      ctx.fillStyle = 'red'
      ctx.fill()
    }
  }, [])

  // Draw a red filled circle on the middle of the canvas
  useEffect(() => {
    // Dummy log necessary to have useEffect re-run when the props change
    console.log('props.maskBoundingBox', props.maskBoundingBox)

    // Draw the circle
    drawCircle()
  }, [drawCircle, props.maskBoundingBox])

  // Redraw the circle when the window size changes
  useWindowSize(drawCircle)

  return (
    <Button
      style={{ backgroundColor: invertMask ? 'white' : 'black', borderColor: invertMask ? 'black' : 'white', borderWidth: 'medium' }}
      onClick={onClick}
      value={value}
    >
      <div
        style={{
          width: buttonSize.width,
          height: buttonSize.height,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          pointerEvents: 'none'
        }}
      >
        <div
          style={{
            padding: '0',
            position: 'relative',
            width: maskBoundingBox.width * maskScaleFactor,
            height: maskBoundingBox.height * maskScaleFactor,
            overflow: 'hidden',
            pointerEvents: 'none'
          }}
        >
          <img
            src={imgSrc}
            alt='Robot camera view'
            style={{
              width: maskBoundingBox.width * maskScaleFactor,
              height: maskBoundingBox.height * maskScaleFactor,
              display: 'block',
              pointerEvents: 'none'
            }}
          />
          <img
            src={maskSrc}
            alt='Mask 0'
            style={{
              position: 'absolute',
              top: '0px',
              filter: invertMask ? 'invert(1)' : 'invert(0)',
              mixBlendMode: invertMask ? 'lighten' : 'multiply',
              width: maskBoundingBox.width * maskScaleFactor,
              height: maskBoundingBox.height * maskScaleFactor,
              display: 'block',
              pointerEvents: 'none'
            }}
          />
          <canvas
            ref={canvasRef}
            width={maskBoundingBox.width * maskScaleFactor}
            height={maskBoundingBox.height * maskScaleFactor}
            style={{
              position: 'absolute',
              top: '0px',
              display: 'block',
              pointerEvents: 'none'
            }}
          />
        </div>
      </div>
    </Button>
  )
}
MaskButton.propTypes = {
  // The button size
  buttonSize: PropTypes.shape({
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired
  }).isRequired,
  // The image source URL
  imgSrc: PropTypes.string.isRequired,
  // The mask source URL
  maskSrc: PropTypes.string.isRequired,
  // Whether or not to invert the mask
  invertMask: PropTypes.bool.isRequired,
  // The scale factor to apply to the mask
  maskScaleFactor: PropTypes.number.isRequired,
  // The bounding box of the mask
  maskBoundingBox: PropTypes.shape({
    x_offset: PropTypes.number.isRequired,
    y_offset: PropTypes.number.isRequired,
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired
  }).isRequired,
  // The function to call when the button is clicked
  onClick: PropTypes.func.isRequired,
  // The value to pass to the onClick function
  value: PropTypes.string.isRequired
}

export default MaskButton
