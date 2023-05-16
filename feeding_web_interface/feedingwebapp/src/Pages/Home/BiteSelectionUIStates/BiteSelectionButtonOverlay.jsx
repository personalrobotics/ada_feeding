// React Imports
import React from 'react'
import Button from 'react-bootstrap/Button'

// Local Imports
import '../Home.css'
import ImageWithButtonOverlay from './ImageWithButtonOverlay'

/**
 * The BiteSelection component appears after the robot has moved above the plate,
 * plate. It enables users to select their desired food item.
 */
const BiteSelectionButtonOverlay = () => {
  /**
   * TODO: Make sure to add the function for Locate Plate and Done Eating
   * Make sure to add back the global state related variable as well!
   */

  // Render the component
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

      {/**
       * TODO: Make sure to add back the callback function calls in onClick
       * property for these buttons! Make sure to also removed the disabled
       * property!
       */}
      <div style={{ display: 'block' }}>
        <Button
          className='doneButton'
          style={{ fontSize: '24px', marginTop: '0px', marginRight: '10px', marginLeft: 'auto', display: 'block' }}
          disabled={true}
        >
          🍽️ Locate Plate
        </Button>
        <Button
          className='doneButton'
          style={{ fontSize: '24px', marginTop: '0px', marginRight: '10px', marginLeft: 'auto', display: 'block' }}
          disabled={true}
        >
          ✅ Done Eating
        </Button>
      </div>
      <div>
        {/**
         * Make sure to subsititute the hard-coded values of imgWidth and imgHeight with the values from
         * constants. The width and height are the dimensions of the image coming from realsense.
         */}
        <ImageWithButtonOverlay
          imgSrc={require('./images/food.jpg')}
          imgWidth={1021}
          imgHeight={779}
          buttonCenters={[
            { name: 'Salad', x: 330, y: 115, rectWidth: 130, rectHeight: 100 },
            { name: 'Salad', x: 490, y: 210, rectWidth: 180, rectHeight: 100 },
            { name: 'Salad', x: 540, y: 170, rectWidth: 130, rectHeight: 100 },
            { name: 'Chicken', x: 250, y: 340, rectWidth: 290, rectHeight: 250 },
            { name: 'Chicken', x: 360, y: 520, rectWidth: 250, rectHeight: 170 },
            { name: 'Chicken', x: 450, y: 620, rectWidth: 200, rectHeight: 140 },
            { name: 'Fries', x: 520, y: 350, rectWidth: 250, rectHeight: 185 },
            { name: 'Fries', x: 750, y: 330, rectWidth: 120, rectHeight: 170 },
            { name: 'Fries', x: 650, y: 640, rectWidth: 150, rectHeight: 100 }
          ]}
        />
      </div>
    </>
  )
}

export default BiteSelectionButtonOverlay
