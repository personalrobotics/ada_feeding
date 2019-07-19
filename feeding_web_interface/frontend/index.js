"use strict";

$(function() {
    $(document).ready(function() {
        // event handlers
        // back buttons
        $("#back_to_food_image_container").click(backToFoodContainer);
        $("#back_to_video_stream_container").click(backToVideoStream);
        // TODO: change this to event handlers for each food
        $("#video_stream").click(showAction);
        // food actions
        var foodActionButtons = document.getElementById("actions").querySelectorAll(".action_btn");
        for (var i = 0; i < foodActionButtons.length; i++) {
            console.log(foodActionButtons[i]);
            foodActionButtons[i].addEventListener("click", performAction);
            foodActionButtons[i].actionName = foodActionButtons[i].innerHTML;
        }

        // get image from backend
        for (var i = 0; i < 6; i++) { 
            var imageDiv = document.createElement("div");
            imageDiv.className = "food_image";
            imageDiv.addEventListener("click", showImage);

            var title = document.createElement("p");
            title.innerHTML = "Apple";
            var image = document.createElement("img");
            image.src = "apple.jpg";
            image.alt = "food";
            image.style.width = "100px";
            image.style.height = "100px";

            imageDiv.appendChild(title);
            imageDiv.appendChild(image);
            $("#food_image_container").append(imageDiv);
        }

        // TODO: get vieo stream from backend, and insert in #video_stream
        
    });

    function backToFoodContainer() {
        $("#food_image_container").css("display", "grid");
        $("#video_stream_container").css("display", "none");
    }

    function showImage() {
        $("#food_image_container").css("display", "none");
        $("#video_stream_container").css("display", "block");
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
});
