// React imports
import './App.css'
import 'bootstrap/dist/css/bootstrap.min.css'
import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { RosConnection } from 'rosreact'

// Local imports
import { useGlobalState, APP_PAGE } from './Pages/GlobalState'
import Header from './Pages/Header/Header'
import Home from './Pages/Home/Home'
import Settings from './Pages/Settings/Settings'
import TestROS from './ros/TestROS'
import BiteSelectionButtonOverlay from './Pages/Home/BiteSelectionUIStates/BiteSelectionButtonOverlay'
import BiteSelectionName from './Pages/Home/BiteSelectionUIStates/BiteSelectionName'
import BiteSelectionPointMask from './Pages/Home/BiteSelectionUIStates/BiteSelectionPointMask'
import RobotVideoStreams from './robot/RobotVideoStreams'

/**
 * Determines what screen to render based on the app page specified in global
 * state.
 *
 * @param {APP_PAGE} appPage - The current app page. Must be one of the
 *        states specified in APP_PAGE.
 * @param {string} rosbridgeURL - The URL of the rosbridge server.
 * @param {bool} debug - Whether to run it in debug mode or not.
 */
function getComponentByAppPage(appPage, rosbridgeURL, webrtcURL, debug) {
  switch (appPage) {
    case APP_PAGE.Home:
      // Must wrap a component in ROS tags for it to be able to connect to ROS
      return (
        <RosConnection url={rosbridgeURL} autoConnect>
          <Header webrtcURL={webrtcURL} />
          <Home debug={debug} webrtcURL={webrtcURL} />
        </RosConnection>
      )
    case APP_PAGE.Settings:
      return (
        <RosConnection url={rosbridgeURL} autoConnect>
          <Header webrtcURL={webrtcURL} />
          <Settings webrtcURL={webrtcURL} />
        </RosConnection>
      )
    default:
      return <div>Invalid app page</div>
  }
}

/**
 * Top-level configuration for the feeding web app. Renders the header and
 * main screen.
 */
function App() {
  // Get the app page
  const appPage = useGlobalState((state) => state.appPage)

  // Get the rosbridge URL
  const rosbridgeURL = 'ws://'.concat(window.location.hostname, ':', process.env.REACT_APP_ROSBRIDGE_PORT)

  // Get the WebRTC signalling server URL
  const webrtcURL = 'http://'.concat(window.location.hostname, ':', process.env.REACT_APP_SIGNALLING_SERVER_PORT)

  // Get the debug flag
  const debug = process.env.REACT_APP_DEBUG === 'true'

  // Render the component
  return (
    <>
      <Router>
        <Routes>
          <Route exact path='/' element={getComponentByAppPage(appPage, rosbridgeURL, webrtcURL, debug)} />
          <Route
            exact
            path='/test_ros'
            element={
              <RosConnection url={rosbridgeURL} autoConnect>
                <TestROS />
              </RosConnection>
            }
          />
          <Route
            exact
            path='/robot'
            element={
              <RosConnection url={rosbridgeURL} autoConnect>
                <RobotVideoStreams webrtcURL={webrtcURL} />
              </RosConnection>
            }
          />
          <Route
            exact
            path='/test_bite_selection_ui/button_overlay_selection'
            element={
              <RosConnection url={rosbridgeURL} autoConnect>
                <Header webrtcURL={webrtcURL} />
                <BiteSelectionButtonOverlay debug={debug} />
              </RosConnection>
            }
          />
          <Route
            exact
            path='/test_bite_selection_ui/point_mask_selection'
            element={
              <RosConnection url={rosbridgeURL} autoConnect>
                <Header webrtcURL={webrtcURL} />
                <BiteSelectionPointMask debug={debug} />
              </RosConnection>
            }
          />
          <Route
            exact
            path='/test_bite_selection_ui/food_name_selection'
            element={
              <RosConnection url={rosbridgeURL} autoConnect>
                <Header webrtcURL={webrtcURL} />
                <BiteSelectionName debug={debug} />
              </RosConnection>
            }
          />
        </Routes>
      </Router>
    </>
  )
}

export default App
