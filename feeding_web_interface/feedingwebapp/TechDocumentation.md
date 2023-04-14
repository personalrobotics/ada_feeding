# Technical Documentation

## Global State vs. Local State

**Local State**: React has the native concept of local state, which exists within a component and triggers re-rendering of the component when changed. Local state should contain the variables that do not need to persist beyond a component (e.g., whether or not a to display a pop-up, countdowns to display within the component, etc.). Use [`useState`](https://react.dev/reference/react/useState) to create and access local state variables; see example [here](https://github.com/personalrobotics/feeding_web_interface/blob/b8f5d970628bbaac43b8b9c7dbc4349ada2f32d7/feedingwebapp/src/Pages/Header/Header.jsx#L22).

**Global State**: Global state are the variables that we need to keep track of across components and page refreshes. This includes what page the app is on (see FAQ below), what stage of the meal the user/robot are currently on, and any settings. All variables in global state should be defined in [`GlobalState.jsx`](https://github.com/personalrobotics/feeding_web_interface/blob/amaln/web_app_revamp/feedingwebapp/src/Pages/GlobalState.jsx), along with their setter functions. Internally, we use [`zustland`](https://github.com/pmndrs/zustand) for global state, which in turn uses cookies to store the state. Therefore, the state will persist across page refreshes, even after clearing the cache. The only way to reset global state is by clearing cookies.

## ROS <--> App Interface (Forthcoming)

## Frequently Asked Questions (FAQ)

Q: Why are we using global state as opposed to a [`Router`](https://www.w3schools.com/react/react_router.asp) to decide which app components to render?
A: If we use a router, each component will have its own URL. That means that by using the "back" button or navigating through the history in their browser, users can reach any screen of the app, even if it doesn't align with the current state of the robot. Since the app's current screen has to be tightly coupled with the state of the robot, we use global state to decide which component to render, as opposed to giving users free reign to jump to any screen of the app by modifying the URL.