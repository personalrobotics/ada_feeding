# Feeding Web Interface

This directory contains the web app that users use to interact with the robot-assisted feeding system. This readme includes an overview and contribution guidelines, while [TechDocumentation.md](https://github.com/personalrobotics/feeding_web_interface/blob/2023PreDeployment/feedingwebapp/TechDocumentation.md) includes more specific documentation of the technical aspects of the app.

## Overview
The overall user flow for this robot can be seen below.

![newWebAppWorkflow](https://user-images.githubusercontent.com/26337328/223597500-5e520b7a-eb2b-45ad-b9e8-91fec1bdeba4.jpg)
(Last Updated 2023/03/07)
<!-- ![Web App State Machine](https://user-images.githubusercontent.com/8277986/191333326-c71a1765-475c-40f6-87da-a79b7c73e0ee.png) 
(Last Updated 2022/09/20) -->

## Dependencies
- [`npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
<!-- - [ROS](http://wiki.ros.org/noetic/Installation) -->

## Getting Started in Computer

### Installation
1. Clone the repo: `git clone git@github.com:personalrobotics/feeding_web_interface.git` using SSH, or `git clone https://github.com/personalrobotics/feeding_web_interface.git` using HTTPS
2. `cd {PATH TO feeding_web_interface}/feedingwebapp`
3. Install all dependencies: `npm install --legacy-peer-deps`

### Usage
1. `cd {PATH TO feeding_web_interface}/feedingwebapp`
2. Start the app: `npm start`
  - Note that if you're not running the robot code alongside the app, set [`debug = true` in `App.jsx`](https://github.com/personalrobotics/feeding_web_interface/blob/b8f5d970628bbaac43b8b9c7dbc4349ada2f32d7/feedingwebapp/src/App.jsx#L17) to be able to move past screens where the app is waiting on the robot. Since the robot is not yet connected, the default is `debug = true`
3. Use a web browser to navigate to `localhost:3000` to see the application.

## Getting Started in Smartphone

### When your phone and a computer both are connected to the same network of PRL 
1. Connect your phone to the same wifi network in the PRL lab that the computer is connected to (e.g., ADA_5G which is used by lab computers like weebo, tobi, ed209). 
2. Run the app in the computer following the above "Getting Started in Computer" instructions. When the app is running in the computer, copy the port number from the app tab's browser and include it after the computer’s IP address followed by a colon (something like `192.xxx.x.x:3000`). The IP addresses of the computers in the PRL lab can be found [here](https://github.com/personalrobotics/pr_docs/wiki/Networking-and-SSH-Information). Depending on which computer is launching the app, use that computer’s IP address in the URL.
3. Then go to that address from the browser in your phone. Then, the app should be running on your phone too. 

### When your phone and a computer are connected to two different networks (only one or neither connected to PRL network)
1. Run the app in the computer following the above "Getting Started in Computer" instructions. 
2. When the app is running in the computer, open another terminal in the computer and install the ngrok agent following [this tutorial](https://ngrok.com/docs/getting-started/#step-2-install-the-ngrok-agent).
3. Then, follow [this tutorial](https://ngrok.com/docs/getting-started/#step-3-connect-your-agent-to-your-ngrok-account) to connect your agent to your ngrok account. Basically, run this command `ngrok config add-authtoken TOKEN` in the terminal with `TOKEN` replaced by the authtoken from your ngrok account.
4. Now, start ngrok by running this command `ngrok http 3000` in the terminal where `3000` is the port number for the app running on the computer with the localhost. 
5. Get part of the Forwarding URL from the console UI shown in your terminal that is before the arrow sign as marked in the image below. Then, run it from your phone browser. You should now see the app ruuning on the phone. For more details on this, follow [this tutorial](https://ngrok.com/docs/getting-started/#step-4-start-ngrok). ![image below](./img/image.png)
6. That URL can be accessed by anyone in the world. You can stop the ngrok agent with `ctrl+c`.

## Contributing

### Documenting Tasks

All tasks should be tracked and documented as [Github Issues](https://github.com/personalrobotics/feeding_web_interface/issues).

### Adding Dependencies
Note that we use `npm`, not `yarn`, to manage dependencies for this project.
- Dependencies added with `npm install ...` should automatically be added when you add `package.json` and `package-lock.json` to your branch.
- Additional dependencies should be documented in this readme.

### Writing Code
- Generally, only work on code if there is a corresponding [Github Issue](https://github.com/personalrobotics/feeding_web_interface/issues) for it.
- Every feature should be on its own branch, and generally only one person should be pushing to one branch.
- Follow the below style guides:
  - [AirBnB React/JSX Style Guide](https://airbnb.io/javascript/react/). This was written before React hooks (e.g, `useState`); see [this for an example of how to order calls to various hooks](https://dev.to/abrahamlawson/react-style-guide-24pp#comment-1f4fd).
  - [AirBnB JavaScript Style Guide](https://airbnb.io/javascript/) for anything not covered in the above guide (e.g., variable naming conventions).
  - [React Styleguidist guide](https://react-styleguidist.js.org/docs/documenting/) for documenting code. Every function and component should be documented, and within functions there should be enough documentation that someone without knowledge of React can understand it.
- Thoroughly test your feature:
  - Run `npm start`, ensure it has no warnings or errors.
  - Thoroughly test your feature, including all edge cases, to ensure it works as expected. This includes trying every combination of buttons/actions, even ones we don't expect users to use, to ensure there are no unaccounted for edge cases.
  - Ensure the [console](https://developer.chrome.com/docs/devtools/console/) has no errors.
  - Thoroughly test responsivity by changing browser/device size to ensure it renders as expected.
- Before creating a Pull Request, run `npm format` and address any warnings or errors.
- Create a Pull Request to merge your branch into `main`. You need at least one approving review to merge.
- Squash all commits on your branch before merging into `main` to ensure a straightforward commit history.