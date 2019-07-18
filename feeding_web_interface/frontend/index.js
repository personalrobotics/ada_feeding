"use strict";

$(function(){
    $(document).ready(function() {
        // add event handlers
        $("#back_to_food_image_container").click(backToFoodContainer);

        // get image from backend, and append
        for (var i = 0; i < 6; i++) { 
            var imageDiv = document.createElement("div");
            imageDiv.className = "food_image";
            imageDiv.addEventListener("click", showImage);

            var title = document.createElement("p");
            title.innerHTML = "Apple";
            var image = document.createElement("img");
            image.src = "apple.jpg";
            image.alt = "food";
            image.style.width = "50px";
            image.style.height = "50px";

            imageDiv.appendChild(title);
            imageDiv.appendChild(image);
            $("#food_image_container").append(imageDiv);
        }
        
    });

    function backToFoodContainer() {
        $("#food_image_container").css("display", "grid");
        $("#video_stream_container").css("display", "none");
    }

    function showImage() {
        $("#food_image_container").css("display", "none");
        $("#video_stream_container").css("display", "block");
    }

});
