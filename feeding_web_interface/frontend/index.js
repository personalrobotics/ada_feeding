"use strict";

let self = this;

// verbosity level of the program
let verbosityLevel = "";
// program mode: user mode (default) / developer mode
let isUserMode = true;
// list of food
let foodImages = ["pics/apple.jpg", "pics/banana.jpg", "pics/candy.jpg", "pics/strawberry.jpg"];
// food image size
let foodImageW = "8em";
let foodImageH = "8em";

$(function() {
    $(document).ready(function() {
        // ROS setup
        self.ros = new ROSLIB.Ros({
            url : 'ws://ada_rosbridge.ngrok.io'
        });
        self.ros.on('error', function(error) {
            console.log('Error connecting to websocket server.');
        });
        self.ros.on('connection', function () {
            console.log("We are connected!");
        });
        self.ros.on('close', function(error) {
            console.error('We lost connection with ROS.');
        });

        // Camera View
        // in user mode
        let userCameraViewer = new MJPEGCANVAS.Viewer({
            divID : "camera",
            host : 'rosvideo.ngrok.io',
            width : 640,
            height : 480,
            topic : "/camera/color/image_raw"
        });
        // in dev mode
        let devCameraViewer = new MJPEGCANVAS.Viewer({
            divID : "video_stream",
            host : 'rosvideo.ngrok.io',
            width : 800,
            height : 480,
            topic : "/camera/color/image_raw"
        });

        // Event handlers
        // verbosity dropdown hover
        $(".dropdown").mouseenter(showDropdown).mouseleave(hideDropdown);
        $(".dropdown").click(showDropdown);
        // verbosity dropdown click
        let verbosityDropdownLists = document.querySelectorAll(".dropdown_content a");
        for (let i = 0; i < verbosityDropdownLists.length; i++) {
            verbosityDropdownLists[i].addEventListener("click", makeVerbositySelection);
        }
        // mode buttons
        $("#user_mode_btn").click(switchMode);
        $("#dev_mode_btn").click(switchMode);
        // back buttons
        $("#back_to_food_image_container").click(backToFoodContainer);
        $("#back_to_video_stream_container").click(backToVideoStream);
        // random food selection
        $("#random_pick_btn").click(chooseFoodRandomly);
        $("#random_image").click(showImage);
        // TODO: change this to event handlers for each food
        $("#video_stream").click(showAction);
        // food actions
        var foodActionButtons = document.getElementById("actions").querySelectorAll(".action_btn");
        for (var i = 0; i < foodActionButtons.length; i++) {
            foodActionButtons[i].addEventListener("click", performAction);
            foodActionButtons[i].actionName = foodActionButtons[i].innerHTML;
        }

        // Get image from backend
        for (var i = 0; i < foodImages.length; i++) { 
            var imageDiv = document.createElement("div");
            imageDiv.className = "food_image";

            let imgaeName = foodImages[i].split("/")[1].split(".")[0];
            // imgaeName = imgaeName.charAt(0).toUpperCase() + imgaeName.substring(1);
            var title = document.createElement("p");
            title.innerHTML = "";
            var image = document.createElement("img");
            image.src = foodImages[i];
            image.alt = imgaeName;
            image.style.width = foodImageW;
            image.style.height = foodImageH;
            // event handler
            image.addEventListener("click", showImage);

            imageDiv.appendChild(title);
            imageDiv.appendChild(image);
            $("#food_image_container").append(imageDiv);
        }        
    });

    function showDropdown() {
        // show the dropdown menu
        $(".dropdown_content").css("display", "block");
    }

    function hideDropdown() {
        // hide the dropdown menu
        $(".dropdown_content").css("display", "none");
    }

    function makeVerbositySelection() {
        hideDropdown();
        // mark the selection
        let selectedVerbosityLevel = this.innerHTML;
        document.getElementById("verbosity_btn").innerHTML = selectedVerbosityLevel;
        // record selected grasp type
        if (selectedVerbosityLevel != "Select verbosity level") {
            verbosityLevel = selectedVerbosityLevel;
            // get verbosity level as integer
            var conversion = convertVerbosity(selectedVerbosityLevel);
            publishVerbosityMsg(conversion);
        } else {
            verbosityLevel = "";
        }
    }

    function convertVerbosity(selectedVerbosityLevel) {
        if (selectedVerbosityLevel == "Basic") {
            return 1;
        } else if (selectedVerbosityLevel == "Intermediate") {
            return 2;
        } else {
            return 3;
        }
    }

    function publishVerbosityMsg(conversion) {
        // publish message to ROS
        var msg_topic = new ROSLIB.Topic({
            ros: ros, 
            name: '/verbosity_msg', 
            messageType: 'std_msgs/Int32'
        });
        msg_topic.advertise();
        var int = new ROSLIB.Message({
            data : conversion
        });
        msg_topic.publish(int);
    }

    function publishFoodItemMsg(foodName) {
        // publish message to ROS
        var msg_topic = new ROSLIB.Topic({
            ros: ros, 
            name: '/foodItem_msg', 
            messageType: 'std_msgs/String'
        });
        msg_topic.advertise();
        var food = new ROSLIB.Message({
            data : foodName
        });
        msg_topic.publish(food);
    }

    function switchMode() {
        if (this.innerHTML == "User Mode") {
            isUserMode = true;
            $("#user_mode_btn").css({"background-color": "#adc7dc", "color": "black"});
            $("#dev_mode_btn").css({"background-color": "#d6e3ed", "color": "white"});
        } else {  // developer mode
            isUserMode = false;
            $("#user_mode_btn").css({"background-color": "#d6e3ed", "color": "white"});
            $("#dev_mode_btn").css({"background-color": "#adc7dc", "color": "black"});
        }
    }

    function backToFoodContainer() {
        if (window.innerWidth < 1000) {
            $("#food_display_container").css("display", "block");
        } else {
            $("#food_display_container").css("display", "grid");
        }
        $("#video_stream_container").css("display", "none");
    }

    function chooseFoodRandomly() {
        let randomImage = document.getElementById("random_image");
        let randomNum = getRandomInt(0, foodImages.length);
        randomImage.src = foodImages[randomNum];
        randomImage.alt = "random image";
    }

    function showImage() {
        // dev mode only
        if (!isUserMode && this.alt != "question mark") {
            $("#food_display_container").css("display", "none");
            $("#video_stream_container").css("display", "block");
        }
        publishFoodItemMsg(this.alt);
    }

    function backToVideoStream() {
        $("#video_stream_container").css("display", "block");
        $("#food_choosing_interface").css("display", "none");
    }

    function showAction() {
        $("#video_stream_container").css("display", "none");
        $("#food_choosing_interface").css("display", "block");
    }

    function performAction(event) {
        // TODO: substitute alert() with the action result sent from backend
        alert("Action: " + event.target.actionName);
    }

    function getRandomInt(min, max) {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
    }
});
