# Technical Documentation

## Global State vs. Local State

**Local State**: React has the native concept of local state, which exists within a component and triggers re-rendering of the component when changed. Local state should contain the variables that need to persist beyond a single rendering of a component, but do not need to persist beyond a component (e.g., whether or not a to display a pop-up, countdowns to display within the component, etc.). Use [`useState`](https://react.dev/reference/react/useState) to create and access local state variables; see example [here](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp/src/Pages/Header/Header.jsx#L22).

**Global State**: Global state are the variables that we need to keep track of across components and page refreshes. This includes what page the app is on (see FAQ below), what stage of the meal the user/robot are currently on, and any settings. All variables in global state should be defined in [`GlobalState.jsx`](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp/src/Pages/GlobalState.jsx), along with their setter functions. Internally, we use [`zustland`](https://github.com/pmndrs/zustand) for global state, which in turn uses cookies to store the state. Therefore, the state will persist across page refreshes, even after clearing the cache. The only way to reset global state is by clearing cookies.

## ROS \<--> App Interface

All functions to interact with ROS are defined in [`ros_helpers.jsx`](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp/src/ros/ros_helpers.jsx). We followed the following guiding principles when making it:

- No other piece of code should need to import `roslib`; all functions that use ROSLIB should be in `ros_helpers.jsx`.
- No other piece of code should need to import `react-ros`. The only exception to that is [`App.jsx`](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp/src/App.jsx) which needs to wrap elements that use ROS in the ROS tag.
- Only `connectToROS()` should call `useROS()`, while other functions should have the `ros` object passed in. This is because hooks can only be used in React components (e.g., not callback functions). So the general idea is that a component that uses ROS will call `let { isConnected, ros } = connectToROS()` within the main code for the component, and will pass `ros` to any subsequent `ros_helpers.jsx` function calls it makes.

`ros_helpers.jsx` currently supports subscribing to and publishing from ROS topics, calling ROS services, and calling ROS actions. Sample code for each of these features can be found in [`TestROS.jsx`](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp/src/ros/TestROS.jsx) and the components it imports, and can be accessed by starting the web app and navigating to `http://localhost:3000/test_ros` in your browser. See [README.md](https://github.com/personalrobotics/feeding_web_interface/tree/main/feedingwebapp/README.md) for more detailed instructions on using the "Test ROS" interface.

### What about getting/setting ROS parameters?

Although the web app's settings menu seem parameter-esque it is not a good idea to update settings by getting/setting ROS parameters. That is because ROS2 parameters are owned by individual nodes, and do not persist beyond the lifetime of a node. Further, we likely want to run downstream code after updating any setting (e.g., writing it to a file as backup), which cannot easily be done when setting ROS parameters. Therefore, **updating settings should be done by ROS service calls, not by getting/setting ROS parameters**. Hence, `ros_helpers.jsx` does not even implement parameter getting/setting.

### Using ROS Services, Actions, etc. Within React

React, by default, will re-render a component any time local state changes. In other words, it will re-run all code in the function that defines that component. However, this doesn't work well for ROS Services, Actions, etc. because: (a) we don't want to keep re-calling the service/action each time the UI re-renders; and (b) we don't want to keep re-creating a service/action client each time the UI re-renders. Therefore, it is crucial to use React Hooks intentionally and to double and triple check that the service/action is only getting once, to avoid bugs.

These resources that can help you understand React Hooks in general:

- https://medium.com/@guptagaruda/react-hooks-understanding-component-re-renders-9708ddee9928
- https://stackoverflow.com/a/69264685

## Frequently Asked Questions (FAQ)

Q: Why are we using global state as opposed to a [`Router`](https://www.w3schools.com/react/react_router.asp) to decide which app components to render?
A: If we use a router, each component will have its own URL. That means that by using the "back" button or navigating through the history in their browser, users can reach any screen of the app, even if it doesn't align with the current state of the robot. Since the app's current screen has to be tightly coupled with the state of the robot, we use global state to decide which component to render, as opposed to giving users free reign to jump to any screen of the app by modifying the URL.
