/*
 * Copyright (c) 2024, Personal Robotics Laboratory
 * License: BSD 3-Clause. See LICENSE.md file in root directory.
 */

import React from 'react'
import ReactDOM from 'react-dom'
import './index.css'
import App from './App'
import reportWebVitals from './reportWebVitals'
import { View } from 'react-native'
import { useWindowSize } from './helpers'

const AppComponent = () => {
  /**
   * NOTE: windowSize.height is necessary as opposed to '100vh' because some
   * browsers have footers (e.g., with control buttons) that are not accounted
   * for in '100vh'.
   */
  let windowSize = useWindowSize()
  return (
    <View style={{ flex: 1, height: windowSize.height }}>
      <App />
    </View>
  )
}

ReactDOM.render(
  <React.StrictMode>
    <AppComponent />
  </React.StrictMode>,
  document.getElementById('root')
)

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals()
