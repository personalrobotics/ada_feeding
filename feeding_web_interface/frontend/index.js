"use strict";

let self = this;

// list of food images
// (note: the lasy entry MUST ALWAYS be the random image!)
let FOODIMAGES = ["pics/food_pics/apple.jpg", "pics/food_pics/cantaloupe.jpg", "pics/food_pics/strawberry.jpg", "pics/food_pics/random.png"];
// list of food action images
let ACTIONIMAGES = ["pics/actions/vertical.jpg", "pics/actions/tilted_vertical.jpg", "pics/actions/tilted_angled.jpg"];
// list of food transfer images
let TRANSFERIMAGES = ["pics/transfers/horizontal.jpg", "pics/transfers/angled.jpg"];
// food image size
let FOOD_IMAGE_W = "15em";
let FOOD_IMAGE_H = "12em";
// action and transfer image size
let COMMON_IMAGE_W = "15em";
let COMMON_IMAGE_H = "11em";

// trial type: 
// 0: Non-autonomous; 1: Autonomous
// 2: Acquisition auto; 3: Timing auto; 4: Transfer auto
let trialType = -1;
// which step is the program currently at?
// 0: init; 1: acquisition; 2: timing; 3: transfer
let currentStep = 0;
// program mode: user mode (default) / developer mode
let isUserMode = true;

// Variables for action client goal
let selectedFoodName = "";
let selectedAcquireAction = -1;

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
        // Action client
        let feedingClient = new ROSLIB.ActionClient({
            ros: self.ros,
            serverName: 'ada_actions',
            actionName: 'ada_demos/ADAFeedingAction'
        });
        // type: 0: acquisition; 1: timing; 2: transfer
        // food
        // action: 0: vertical; 1: tilted vertical; 2: tilted angled
        let feedingGoal = new ROSLIB.Goal({
            actionClient: feedingClient,
            goalMessage: {
                type: currentStep - 1,
                food: selectedFoodName,
                action: selectedAcquireAction
            }
        });
        feedingGoal.on('result', handleFeedingResult);

        // ROS topic
        // publish
        self.trialTypeTopic = new ROSLIB.Topic({
            ros: self.ros, 
            name: '/trial_type', 
            messageType: 'std_msgs/Int32'
        });
        self.foodItemTopic = new ROSLIB.Topic({
            ros: self.ros, 
            name: '/foodItem_msg', 
            messageType: 'std_msgs/String'
        });
        // subscribe
        self.statusTopic = new ROSLIB.Topic({
            ros: self.ros, 
            name: '/status', 
            messageType: 'std_msgs/Status'
        });
        self.statusTopic.subscribe(handleStatus);
        self.actionDoneTopic = new ROSLIB.Topic({
            ros: self.ros, 
            name: '/action_done', 
            messageType: 'std_msgs/String'
        });
        self.actionDoneTopic.subscribe(handleActionDone);
        self.timingDoneTopic = new ROSLIB.Topic({
            ros: self.ros, 
            name: '/timing_done', 
            messageType: 'std_msgs/String'
        });
        self.timingDoneTopic.subscribe(handleTimingDone);
        self.transferDoneTopic = new ROSLIB.Topic({
            ros: self.ros, 
            name: '/transfer_done', 
            messageType: 'std_msgs/String'
        });
        self.transferDoneTopic.subscribe(handleTransferDone);

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
        // trial type dropdown hover
        $(".dropdown").mouseenter(showDropdown).mouseleave(hideDropdown);
        $(".dropdown").click(showDropdown);
        // trial type dropdown click
        let trialDropdownLists = document.querySelectorAll(".dropdown_content a");
        for (let i = 0; i < trialDropdownLists.length; i++) {
            trialDropdownLists[i].addEventListener("click", makeTrialSelection);
        }
        // mode buttons
        $("#user_mode_btn").click(switchMode);
        $("#dev_mode_btn").click(switchMode);
        // feed button
        $("#time_to_feed_btn").click(feedNow);
        // restart button
        $("#restart_btn").click(restart);
        // back buttons
        $("#back_to_food_image_container").click(backToFoodContainer);
        $("#back_to_video_stream_container").click(backToVideoStream);
        // TODO: change this to event handlers for each food
        $("#video_stream").click(showAction);
        // food actions
        let foodActionButtons = document.getElementById("actions").querySelectorAll(".action_btn");
        for (let i = 0; i < foodActionButtons.length; i++) {
            foodActionButtons[i].addEventListener("click", performAction);
            foodActionButtons[i].actionName = foodActionButtons[i].innerHTML;
        }
        // emergency stop
        $("#estop").click(estop);
        // images
        for (let i = 0; i < 3; i++) {
            createAndAddImages(i);
        }

        // hide all the status bar
        $(".status").css("display", "none");

        //////////////////////////////////////////////////////////////
        $("#test1").click(handleActionDone);
        $("#test2").click(handleTimingDone);
        $("#test3").click(handleTransferDone);
    });

    function showDropdown() {
        // show the dropdown menu
        $(".dropdown_content").css("display", "block");
    }

    function hideDropdown() {
        // hide the dropdown menu
        $("#trial_dropdown_container").css("display", "none");
    }

    function makeTrialSelection() {
        hideDropdown();
        // mark the selection
        let selectedTrialType = this.innerHTML;
        document.getElementById("trial_btn").innerHTML = selectedTrialType;
        // record selected trial type
        if (selectedTrialType != "Please select") {
            trialType = parseInt(this.id);
            // publish trial type message to ROS
            let type = new ROSLIB.Message({
                data : trialType
            });

            // -----------------------------------------------
            self.trialTypeTopic.publish(type);

        } else {
            trialType = -1;
        }
    }

    function switchMode() {
        if (this.innerHTML === "User Mode") {
            isUserMode = true;
            $("#user_mode_btn").css({"background-color": "#adc7dc", "color": "black"});
            $("#dev_mode_btn").css({"background-color": "#d6e3ed", "color": "white"});
            // restart the whole program
            restart();
        } else {  // developer mode
            isUserMode = false;
            $("#user_mode_btn").css({"background-color": "#d6e3ed", "color": "white"});
            $("#dev_mode_btn").css({"background-color": "#adc7dc", "color": "black"});
            // hide trial type selection
            $("#trial_dropdown_container").css("display", "none");
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

    function handleStatus(msg) {
        let type = msg.type;
        let data = msg.data;
        // display the response
        document.getElementById(type + "_status").innerHTML = data;
    }

    function makeFoodSelection() {
        if (trialType < 0 && isUserMode) {
            alert("Please select a trial type!");
        } else {
            let foodName = this.alt;
            // pick a food if in random selection
            if (foodName === "random") {
                // the last entry in FOODIMAGES is always the random image
                // so we should exclude the last index when generating a random number
                let randomNum = getRandomInt(0, FOODIMAGES.length - 1);
                foodName = FOODIMAGES[randomNum].split("/")[2].split(".")[0];
            }
            // record the selected food name
            selectedFoodName = foodName;

            // // --------------------------------------------------------------
            // self.foodItemTopic.publish(foodName);
            publishFoodItemMsg(selectedFoodName);

            // continue to the next step
            if (isUserMode) {
                // show action page
                handleFoodSelectionDone();
                // hide trial type selection
                $("#trial_dropdown_container").css("display", "none");
                // disable the developer mode button
                document.getElementById("dev_mode_btn").disabled = true;
            } else {  // show large camera view in dev mode only
                $("#food_display_container").css("display", "none");
                $("#video_stream_container").css("display", "block");
            }
        }
    }

    function handleFoodSelectionDone() {
        currentStep++;
        // show the action page
        $("#food_pick_container").css("display", "none");
        $("#action_pick_container").css("display", "block");
        if (trialType === 0 || trialType === 3 || trialType === 4) {
            // let user pick an action
            document.querySelector("#action_pick_container h2").innerHTML = "Choose an action to pick up the food";
            // show all the action images
            $(".auto_msg").css("display", "none");
            $("#action_pick_container img, .action_name").css("display", "block");
            // hide status bar
            $("#action_status").css("display", "none");
        } else {
            // auto
            document.querySelector("#action_pick_container h2").innerHTML = "Please wait...";
            // hide all the action images
            $(".auto_msg").css("display", "block");
            $("#action_pick_container img, .action_name").css("display", "none");
            // display status bar
            showStatusBar(true, "action");
            // // -----------------------------------------------
            // // send the goal
            // feedingGoal.send();
        }
    }

    function makeActionSelection() {
        // // ------------------------------------------------
        // // send the goal
        // if (this.alt === "vertical") {
        //     selectedAcquireAction = 0;
        // } else if (this.alt === "tilted vertical") {
        //     selectedAcquireAction = 1;
        // } else {  // "tilted angled"
        //     selectedAcquireAction = 2;
        // }
        // feedingGoal.send();

        // display status bar
        showStatusBar(false, "action");
    }

    function handleActionDone(msg) {
        currentStep++;
        // show the timing page
        $("#action_pick_container").css("display", "none");
        $("#timing_pick_container").css("display", "block");
        if (trialType === 0 || trialType === 2 || trialType === 4) {
            // let user decide when the robot should feed
            document.querySelector("#timing_pick_container h2").innerHTML = 
                    "Choose when to approach";
            // show the feed button
            $(".auto_msg").css("display", "none");
            $("#time_to_feed_btn").css("display", "block");
            // hide status bar
            $("#timing_status").css("display", "none");
        } else {
            // auto
            document.querySelector("#timing_pick_container h2").innerHTML = "Please wait...";
            // hide the feed button
            $(".auto_msg").css("display", "block");
            $("#time_to_feed_btn").css("display", "none");
            // display status bar
            showStatusBar(false, "timing");
            // // --------------------------------------
            // // send the goal
            // feedingGoal.send();
        }
    }

    function feedNow() {
        // // ------------------------------------------------
        // // send the goal
        // feedingGoal.send();

        // display status bar
        showStatusBar(false, "timing");
    }
    
    function handleTimingDone(msg) {
        currentStep++;
        // show the transfer page
        $("#timing_pick_container").css("display", "none");
        $("#transfer_pick_container").css("display", "block");
        if (trialType === 0 || trialType === 2 || trialType === 3) {
            // let user decide how the food should be transfered
            document.querySelector("#transfer_pick_container h2").innerHTML = "Choose how to feed";
            // show the transfer images
            $(".auto_msg").css("display", "none");
            $("#transfer_pick_container img, .transfer_name").css("display", "block");
            // hide status bar
            $("#transfer_status").css("display", "none");
        } else {
            // auto
            document.querySelector("#transfer_pick_container h2").innerHTML = "Please wait...";
            // hide the transfer images
            $(".auto_msg").css("display", "block");
            $("#transfer_pick_container img, .transfer_name").css("display", "none");
            // display status bar
            showStatusBar(true, "transfer");
            // // --------------------------------------
            // // send the goal
            // feedingGoal.send();
        }
    }

    function makeTransferSelection() {
        // // --------------------------------------
        // // send the goal
        // feedingGoal.send();

        // display status bar
        showStatusBar(true, "transfer");
    }
    
    function handleTransferDone(msg) {
        // show the restart page
        $("#transfer_pick_container").css("display", "none");
        $("#restart_pick_container").css("display", "block");
    }

    function restart() {
        // show front page (food selection page)
        $("#restart_pick_container").css("display", "none");
        $("#food_pick_container").css("display", "block");
        // enable the developer mode button
        document.getElementById("dev_mode_btn").disabled = false;
        // reset variables
        trialType = -1;
        currentStep = 0;
        isUserMode = true;
        selectedFoodName = "";
        selectedAcquireAction = -1;
        // show trial type selection
        $("#trial_dropdown_container").css("display", "block");
        document.getElementById("trial_btn").innerHTML = "Please select";
    }

    function estop() {
        if (trialType < 0 && isUserMode) {
            alert("Please select a trial type!");
        } else {
            // TODO: publish estop message

        }
    }

    function handleFeedingResult(result) {
        let typeId = result.type;
        let success = result.success;
        if (typeId === 1) {
            $("#timing_status").html("Approaching");
        } else if (typeId === 2) {
            $("#transfer_status").html("Ready to eat");
        } else {  // acquisition
            document.getElementById("action_status").innerHTML = success ? "SUCCEEDED" : "FAILED";
        }
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

    // Helper functions
    // create image elements and add them to the DOM
    // imageType: 0 (food), 1 (action), 2 (transfer)
    function createAndAddImages(imageType) {
        let imageList = FOODIMAGES;
        let type = "food";
        if (imageType === 1) {
            imageList = ACTIONIMAGES;
            type = "action";
        } else if (imageType === 2) {
            imageList = TRANSFERIMAGES;
            type = "transfer";
        }
        let mainContainer = $("#"+ type + "_image_container");
        for (let i = 0; i < imageList.length; i++) { 
            // create DOM elements
            let imageDiv = document.createElement("div");
            imageDiv.className = type + "_image";
            let imgaeName = imageList[i].split("/")[2].split(".")[0].replace("_", " ").toLowerCase();
            if (imageType != 0) {
                // add a text to display the image name
                let text = document.createElement("p");
                text.className = type + "_name";
                text.innerHTML = imgaeName.charAt(0).toUpperCase() + imgaeName.substring(1, imgaeName.length);
                imageDiv.appendChild(text);
            }
            let image = document.createElement("img");
            image.src = imageList[i];
            image.alt = imgaeName;  // all lowercase
            if (imageType != 0) {
                image.style.width = COMMON_IMAGE_W;
                image.style.height = COMMON_IMAGE_H;
            } else {
                image.style.width = FOOD_IMAGE_W;
                image.style.height = FOOD_IMAGE_H;
            }
            // event handler
            if (imageType === 0) {
                image.addEventListener("click", makeFoodSelection);
            } else if (imageType === 1) {
                image.addEventListener("click", makeActionSelection);
            } else {
                image.addEventListener("click", makeTransferSelection);
            }
            // add to DOM
            imageDiv.appendChild(image);
            mainContainer.append(imageDiv);
        }
        // add a status displayer
        if (imageType != 0) {
            let statusBar = document.createElement("p");
            statusBar.id = type + "_status";
            statusBar.className = "status";
            statusBar.innerHTML = "Waiting for robot response...";
            mainContainer.append(statusBar);
        }
    }

    // --------------------------------------------------------------------------
    function publishFoodItemMsg(foodName) {
        // publish food item message to ROS
        let food = new ROSLIB.Message({
            data: foodName
        });
        self.foodItemTopic.publish(food);
    }

    function showStatusBar(expandLayout, name) {
        let statusBar = $("#" + name + "_status");
        statusBar.css("display", "block");
        statusBar.html("Waiting for robot response...");
        if (expandLayout) {
            // the status bar should take 2 columns
            statusBar.css({
                gridColumnStart: "1",
                gridColumnEnd: "3",
                marginBottom: "3em"
            });
        }
    }

    function getRandomInt(min, max) {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
    }
});
