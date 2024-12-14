/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

// React imports
import React from 'react'
// PropTypes is used to validate that the used props are in fact passed to this Component
import PropTypes from 'prop-types'
import styled from 'styled-components'

/**
 * Configure the styles for a group of buttons where only one is active at a time.
 *
 * NOTE: Over here we use `styled-components` whereas elsewhere in the app, e.g.,
 * Home.css and App.css, we use separate CSS files. Eventually we should pick one
 * and stick with it. The below blog post has some tradeoffs:
 * https://getstream.io/blog/styled-components-vs-css-stylesheets/
 */
const Button = styled.button`
  background-color: purple;
  color: white;
  font-size: 25px;
  padding: 10px 20px;
  border-radius: 5px;
  margin: 10px 0px;
  cursor: pointer;
`
const ButtonToggle = styled(Button)`
  opacity: 0.6;
  ${({ active }) =>
    active &&
    `
    opacity: 1;
  `}
`
const ButtonGroup = styled.div`
  display: flex;
`

/**
 * Takes in the global state for a particular setting and the values that
 * setting can take. Returns the toggleable group of buttons, configured with
 * the right callbacks to change the global state.
 *
 * @param {array} valueOptions - the values that this setting can take on. A list
 *        of strings.
 * @param {string} currentValue - the current value of this setting. Must be in
 *        `valueOptions`
 * @param {array} callbackFunctions - the callback functions to call when the
 *       setting is changed. Must be the same length as `valueOptions`. The callback
 *       function will be called with the new value of the setting.
 * @param {boolean} horizontal - whether to display the buttons horizontally
 *       or vertically. Defaults to true.
 */
function ToggleButtonGroup(props) {
  let valueOptions = props.valueOptions
  let currentValue = props.currentValue
  let callbackFunctions = props.callbackFunctions
  return (
    <ButtonGroup
      style={{
        flexDirection: props.horizontal ? 'row' : 'column'
      }}
    >
      {valueOptions.map((value, i) => (
        <ButtonToggle key={value} active={currentValue === value} onClick={() => callbackFunctions[i](value)}>
          {value}
        </ButtonToggle>
      ))}
    </ButtonGroup>
  )
}
ToggleButtonGroup.propTypes = {
  valueOptions: PropTypes.array.isRequired,
  currentValue: PropTypes.string.isRequired,
  callbackFunctions: PropTypes.array.isRequired,
  horizontal: PropTypes.bool
}
ToggleButtonGroup.defaultProps = {
  horizontal: true
}

export default ToggleButtonGroup
