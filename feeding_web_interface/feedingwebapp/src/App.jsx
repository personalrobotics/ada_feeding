// React imports
import './App.css'
import React from 'react'
import 'bootstrap/dist/css/bootstrap.min.css'
import { ROS } from 'react-ros'

// Local imports
import { useGlobalState, APP_PAGE } from './Pages/GlobalState'
import Header from './Pages/Header/Header'
import Home from './Pages/Home/Home'
import Settings from './Pages/Settings/Settings'

// Flag for whether to run the app in debug mode or not. When running in debug
// mode, the app does not connect to ROS. Anytime where it would wait for the
// robot to finish an action before continuing, the app instead displays a
// button to continue.
const debug = true

/**
 * Determines what screen to render based on the app page specified in global
 * state.
 *
 * @param {APP_PAGE} appPage - The current app page. Must be one of the
 *        states specified in APP_PAGE.
 * @param {bool} debug - Whether to run it in debug mode or not.
 */
function getComponentByAppPage(appPage, debug) {
  switch (appPage) {
    case APP_PAGE.Home:
      // Must wrap a component in ROS tags for it to be able to connect to ROS
      return (
        <ROS>
          <Home debug={debug} />
        </ROS>
      )
    case APP_PAGE.Settings:
      return <Settings />
  }
}

/**
 * Top-level configuration for the feeding web app. Renders the header and
 * main screen.
 */
function App() {
  // Get the app page
  const appPage = useGlobalState((state) => state.appPage)

  // Render the component
  return (
    <>
      <Header />
      {getComponentByAppPage(appPage, debug)}
    </>
  )
}

export default App
