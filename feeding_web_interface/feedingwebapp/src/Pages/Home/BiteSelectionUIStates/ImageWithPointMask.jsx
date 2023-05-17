// React Imports
import React, { useState } from 'react'
import Button from 'react-bootstrap/Button'
import Row from 'react-bootstrap/Row'
import { scaleWidthHeightToWindow } from '../../../helpers'
import PropTypes from 'prop-types'

import '../Button.css'

/**
 * Displays the plate pictures and allows users to select food items by selecting a
 * point on the image
 *
 * @param {string} imgSrc - The local filepath for where the image is located
 * @param {number} imgWidth - The width of the image that is being passed in from
 *        the realsense camera
 * @param {number} imgHeight - The height of the image that is being passed in from
 *        the realsense camera
 */
const ImageWithPointMask = (props) => {
  const width = scaleWidthHeightToWindow(props.imgWidth, props.imgHeight, 0, 0, 0, 0).width
  const height = scaleWidthHeightToWindow(props.imgWidth, props.imgHeight, 0, 0, 0, 0).height
  const [foodMasksToDisplay, setFoodMasksToDisplay] = useState([])

  const imageClicked = (event) => {
    let rect = event.target.getBoundingClientRect()
    let x = event.clientX - rect.left
    let y = event.clientY - rect.top
    let masks = []
    if (y < height / 2 - 15) {
      // Salad
      masks.push(
        <img
          key={1}
          style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
          className='RectButtonsOnImg'
          src={require('./images/salad_mask1.jpg')}
          alt='The first contender mask found by the food detection algorithm'
        />
      )
      masks.push(
        <img
          key={2}
          style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
          className='RectButtonsOnImg'
          src={require('./images/salad_mask2.jpg')}
          alt='The second contender mask found by the food detection algorithm'
        />
      )
      masks.push(
        <img
          key={3}
          style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
          className='RectButtonsOnImg'
          src={require('./images/salad_mask3.png')}
          alt='The third contender mask found by the food detection algorithm'
        />
      )
    } else {
      if (x < width / 2 + 35) {
        // Chicken
        masks.push(
          <img
            key={1}
            style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
            className='RectButtonsOnImg'
            src={require('./images/chicken_mask1.jpg')}
            alt='The first contender mask found by the food detection algorithm'
          />
        )
        masks.push(
          <img
            key={2}
            style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
            className='RectButtonsOnImg'
            src={require('./images/chicken_mask2.jpg')}
            alt='The second contender mask found by the food detection algorithm'
          />
        )
        masks.push(
          <img
            key={3}
            style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
            className='RectButtonsOnImg'
            src={require('./images/chicken_mask3.jpg')}
            alt='The third contender mask found by the food detection algorithm'
          />
        )
      } else {
        // fries
        masks.push(
          <img
            key={1}
            style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
            className='RectButtonsOnImg'
            src={require('./images/fries_mask1.jpg')}
            alt='The first contender mask found by the food detection algorithm'
          />
        )
        masks.push(
          <img
            key={2}
            style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
            className='RectButtonsOnImg'
            src={require('./images/fries_mask2.jpg')}
            alt='The second contender mask found by the food detection algorithm'
          />
        )
        masks.push(
          <img
            key={3}
            style={{ width: width / 3 - 2, border: 'solid', margin: '1px' }}
            className='RectButtonsOnImg'
            src={require('./images/fries_mask3.jpg')}
            alt='The third contender mask found by the food detection algorithm'
          />
        )
      }
    }
    setFoodMasksToDisplay(masks)
  }

  return (
    <>
      <div style={{ position: 'relative', top: '0', left: '0' }}>
        <div style={{ position: 'fixed', top: '120px', left: '25px' }}>
          <h1 style={{ background: '#FFC107' }}>Option B</h1>
        </div>
        <img
          onClick={imageClicked}
          src={props.imgSrc}
          style={{ position: 'absolute', top: '0', left: '0', width: width, height: height }}
          alt="The view from the robot's camera, showing the plate of food in front of the robot"
        />
      </div>
      <div style={{ position: 'relative', top: height, left: 0 }}>
        <Row>{foodMasksToDisplay.map((img) => img)}</Row>
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
          <a style={{ textDecoration: 'None', color: 'white' }} href='/test_bite_selection_ui/food_name_selection'>
            Next
          </a>
        </Button>
      </div>
    </>
  )
}

ImageWithPointMask.propTypes = {
  imgSrc: PropTypes.string.isRequired,
  imgWidth: PropTypes.number.isRequired,
  imgHeight: PropTypes.number.isRequired
}

export default ImageWithPointMask
