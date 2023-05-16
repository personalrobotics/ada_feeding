// React imports
import './App.css'
import 'bootstrap/dist/css/bootstrap.min.css'
import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ROS } from 'react-ros'

// Local imports
import { useGlobalState, APP_PAGE } from './Pages/GlobalState'
import Header from './Pages/Header/Header'
import Home from './Pages/Home/Home'
import Settings from './Pages/Settings/Settings'
import TestROS from './ros/TestROS'
import BiteSelectionButtonOverlay from './Pages/Home/BiteSelectionUIStates/BiteSelectionButtonOverlay'
import BiteSelectionName from './Pages/Home/BiteSelectionUIStates/BiteSelectionName'
import BiteSelectionPointMask from './Pages/Home/BiteSelectionUIStates/BiteSelectionPointMask'

// Flag for whether to run the app in debug mode or not. When running in debug
// mode, the app does not connect to ROS. Anytime where it would wait for the
// robot to finish an action before continuing, the app instead displays a
// button to continue.
const debug = false

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
          <Header />
          <Home debug={debug} />
        </ROS>
      )
    case APP_PAGE.Settings:
      return (
        <ROS>
          <Header />
          <Settings />
        </ROS>
      )
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
      <Router>
        <Routes>
          <Route exact path='/' element={getComponentByAppPage(appPage, debug)} />
          <Route
            exact
            path='/test_ros'
            element={
              <ROS>
                <TestROS />
              </ROS>
            }
          />
          <Route
            exact
            path='/test_bite_selection_ui/button_overlay_selection'
            element={
              <ROS>
                <Header />
                <BiteSelectionButtonOverlay debug={debug} />
              </ROS>
            }
          />
          <Route
            exact
            path='/test_bite_selection_ui/point_mask_selection'
            element={
              <ROS>
                <Header />
                <BiteSelectionPointMask debug={debug} />
              </ROS>
            }
          />
          <Route
            exact
            path='/test_bite_selection_ui/food_name_selection'
            element={
              <ROS>
                <Header />
                <BiteSelectionName debug={debug} />
              </ROS>
            }
          />
        </Routes>
      </Router>
    </>
  )
}

export default App
