// React Imports
import React from 'react'
import Button from 'react-bootstrap/Button'
import { scaleWidthHeightToWindow } from '../../../helpers'
import PropTypes from 'prop-types'

import '../Button.css'

/**
 * Displays the plate pictures and allows users to select food items by selecting buttons
 * overlaid on the screen
 *
 * @param {string} imgSrc - The local filepath for where the image is located
 * @param {number} imgWidth - The width of the image that is being passed in from
 *        the realsense camera
 * @param {number} imgHeight - The height of the image that is being passed in from
 *        the realsense camera
 * @param {arrayOf({name, x, y, rectWidth, rectHeight})} buttonCenters - Each button
 *        that needs to be overlaid on the image with its corresponding name, (x, y)
 *        coordinates of the button's top-left corner and then the rectangle button's
 *        width and height
 * @returns
 */

const ImageWithButtonOverlay = (props) => {
  const width = scaleWidthHeightToWindow(props.imgWidth, props.imgHeight, 0, 0, 0, 0).width
  const height = scaleWidthHeightToWindow(props.imgWidth, props.imgHeight, 0, 0, 0, 0).height
  const scaleFactor = scaleWidthHeightToWindow(props.imgWidth, props.imgHeight, 0, 0, 0, 0).scaleFactor

  return (
    <>
      <div style={{ position: 'relative', top: '0', left: '0' }}>
        <div style={{ position: 'fixed', top: '120px', left: '25px' }}>
          <h1 style={{ background: '#FFC107' }}>Option A</h1>
        </div>
        <img src={props.imgSrc} style={{ position: 'absolute', top: '0', left: '0', width: width, height: height }} />
        {props.buttonCenters.map((location, i) => {
          let xVal = location.x * scaleFactor
          let yVal = location.y * scaleFactor
          return (
            <svg
              key={i}
              style={{
                position: 'absolute',
                top: yVal,
                left: xVal,
                width: location.rectWidth * scaleFactor,
                height: location.rectHeight * scaleFactor,
                zIndex: 1
              }}
            >
              <svg width={width} height={height}>
                <rect
                  x={0}
                  y={0}
                  rx='5'
                  ry='5'
                  width={location.rectWidth * scaleFactor}
                  height={location.rectHeight * scaleFactor}
                  fillOpacity={0.2}
                  stroke='black'
                  strokeWidth={5}
                  className='RectButtonsOnImg'
                />
              </svg>
            </svg>
          )
        })}
      </div>
      <div style={{ position: 'relative', top: height, left: 0 }}>
        <Button
          variant='secondary'
          className='mx-1 mb-1'
          style={{
            paddingLeft: '20px',
            paddingRight: '20px',
            marginLeft: '0px',
            marginRight: '0px',
            fontSize: '25px',
            position: 'fixed',
            right: '0px',
            bottom: '0px'
          }}
          size='lg'
        >
          <a style={{ textDecoration: 'None', color: 'white' }} href='/test_bite_selection_ui/point_mask_selection'>
            Next
          </a>
        </Button>
      </div>
    </>
  )
}

ImageWithButtonOverlay.propTypes = {
  imgSrc: PropTypes.string.isRequired,
  imgWidth: PropTypes.number.isRequired,
  imgHeight: PropTypes.number.isRequired,
  buttonCenters: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      rectWidth: PropTypes.number.isRequired,
      rectHeight: PropTypes.number.isRequired
    })
  ).isRequired
}

export default ImageWithButtonOverlay
