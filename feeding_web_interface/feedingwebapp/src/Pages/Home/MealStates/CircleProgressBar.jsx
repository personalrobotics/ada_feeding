/*
 * Copyright (c) 2024-2025, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// Import circle progress bar from progress bar package
import { Circle } from './progressbar.js'
// React imports
import React, { useEffect, useState } from 'react'
import { useMediaQuery } from 'react-responsive'
import { View } from 'react-native'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'

/**
 * A functional component to render full circle progress bars
 * in robot moving screens. It is imported in RobotMotion.jsx.
 *
 * @param {number} proportion - the current progress level of robot's motion
 */
export default function CircleProgressBar(props) {
  // A local state variable to keep track of progress bar creation
  const [bar, setBar] = useState(null)
  // Flag to check if the current orientation is portrait
  const isPortrait = useMediaQuery({ query: '(orientation: portrait)' })
  // Size variables for progressbar (width, height, fontsize) in portrait and landscape
  let circleWidth = isPortrait ? '90%' : null
  let circleHeight = isPortrait ? null : '90%'
  let textFontSize = isPortrait ? '8vh' : '14vh'

  // useEffect React Hook is used to synchronize with RobotMotion.jsx data to render circle progress bar
  useEffect(() => {
    // Effect: if circle progress bar does not exist, create it; else render it with text
    if (bar === null) {
      setBar(
        new Circle('#container', {
          // Green color indicating smooth progression
          color: '#008000',
          // Width of the stroke to be applied to the shape
          strokeWidth: 4,
          // Width of the trail to be applied to the shape before stroke fills in
          trailWidth: 1,
          // Specify centered text style for circle progress bar
          text: {
            style: {
              fontFamily: '"Raleway", Helvetica, sans-serif',
              fontSize: textFontSize,
              position: 'absolute',
              left: '50%',
              top: '50%',
              padding: '0px',
              margin: '0px',
              transform: 'translate(-50%, -50%)'
            }
          }
        })
      )
    } else {
      // Sets progress instantly without animation and clears all animations for path.
      bar.set(props.proportion)
      // Sets text to given a string which is motion progress percentage here.
      bar.setText(Math.round(props.proportion * 100) + '%')
      // Set the text font size. Necessary since this changes upon orientation change.
      bar.text.style.fontSize = textFontSize
    }
    // Fverytime items in dependency array (the second argument) update, useEffect runs.
  }, [setBar, bar, props.proportion, textFontSize])
  // Render HTML in a view flexbox through RobotMotion.jsx.
  return (
    <View
      id='container'
      style={{
        flex: 1,
        margin: '15px',
        width: isPortrait ? circleWidth : circleHeight,
        justifyContent: 'center'
      }}
    ></View>
  )
}

// Progress proportion corresponding with the motion the robot is executing.
CircleProgressBar.propTypes = { proportion: PropTypes.number.isRequired }
