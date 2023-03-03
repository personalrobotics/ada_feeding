### How to Run Plate Locator Algorithm
Open up 4 terminals in your computer (we assume you are using weebo that is connected to ADA robot in lab) and in each terminal run the following commands (1-4) after successfully building your workspace by respectively doing: `cd <catkin_ws>; catkin build; . devel/setup.bash`

1. `roscore`
2. `roslaunch ada_feeding rviz.launch`
3. `python ada_feeding/scripts/run_plate_locator.py`
4. `roslaunch ada_feeding feeding.launch treeFile:=plate_locator.xml sim:=false use_forque:=false`

Now, open up a fifth terminal and go to the directory of [feeding web app](https://github.com/personalrobotics/feeding_web_interface/tree/2022_revamp/feedingwebapp). From there run `npm start` which should start the app in your computer. If `npm` was not installed in your computer, run `npm install` before the `npm start` command.

**Start App on Your Phone:** Connect your phone to the same wifi of your computer (in this case ADA_5G which is used by weebo). As the app is already running in weebo. Copy the url from the app tab's browser (something like `192.168.2.22:3000` which is weebo's IP address followed by port number) to browser in your phone. Then, the app should be running in your phone too. Go to the plate locator buttons by clicking on `Start Feeding` button from app's homepage.

### Different Components of Plate Locator Algorithm
1. The `run_plate_locator.py` script contains a ros subscriber and a ros service that respectively processes ros images from rviz camera topic to cv image. Then, the service uses that cv image to see if partial plate is detected, and if so, it computes movement direction for full plate detection.
2. The `plate_locator.xml` runs the behavior tree that listens to ros msgs published by the app and moves the robot arm accordingly.
3. The `Home.js` file has all the ros topics and their publishers defined.