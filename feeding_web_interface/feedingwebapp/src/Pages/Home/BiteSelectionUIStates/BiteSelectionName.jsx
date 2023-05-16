// React Imports
import React from 'react'
import Button from 'react-bootstrap/Button'

// Local Imports
import '../Home.css'
import ImageWithButtonName from './ImageWithButtonName'

/**
 * The BiteSelection component appears after the robot has moved above the plate,
 * plate. It enables users to select their desired food item.
 */
const BiteSelectionName = () => {
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
        <ImageWithButtonName
          imgSrc={require('./images/food.jpg')}
          imgWidth={1021}
          imgHeight={779}
          foodItems={['Chicken', 'Salad', 'Fries']}
        />
      </div>
    </>
  )
}

export default BiteSelectionName
