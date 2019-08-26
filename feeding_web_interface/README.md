# feeding_web_interface

1. ssh into nano to start the camera node, make sure you export ROS_MASTER_URI to LOVELACE

    ```ssh nano@192.168.2.121``` passcode is normal lab passcode
    
    ```export ROS_MASTER_URI=https://192.168.2.145:11311```
    
    ```./run_camser.sh```
2. launch rosbridge_server to create localhost with the following command

    ```roslaunch rosbridge_server rosbridge_websocket```
    
3. enter the following command:

    ```python -m SimpleHTTPServer 8082```
    
4. go to ws/src/feeding_web_interface/frontend

    ```./ngrok start -all``` to start proxy server
    
5. go to webpage http://ada_feeding.ngrok.io/ to see the demo web interface. If you follow the above steps carefullt, you should now be able to see the camera stream displated on the right and the foodImage on left.
