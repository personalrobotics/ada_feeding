import cv2
import numpy as np
import math

# This will work because it DOES pick out blue color and uses that grayscale or black-and-white image in mask

def get_offset_from_partial_plate_view():
    # load an image using 'imread'
    original_image = cv2.imread("/Users/raidakarim/Downloads/half_blue_plate.png")
    (h, w) = original_image.shape[:2]
    # where w//2, h//2 are the required frame/image centeroid's XYcoordinates.
    centerX = w // 2
    centerY = h // 2
    #print('centerX', centerX)
    #print('centerY', centerY)
    # original_image = cv2.imread("/Users/raidakarim/Downloads/full_blue_plate.png")

    # convert images from BGR (Blue, Green, Red) to HSV (Hue-- color, Saturation-- density, Value-- lightness)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    ### blue color detection and masking ###
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    # define range of blue color in HSV here to create a mask of blue colored object
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # convert the mask image to binary image
    # The method returns two outputs.
    # The first is the threshold that was used and the second output is the thresholded image.
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    # ret, thresh = cv2.threshold(mask, 127, 255, 0)

    # calculate moments of binary image
    m = cv2.moments(mask)
    #print(m)

    # calculate x,y coordinates of center
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])

    center = [centerX, centerY]
    currPoint = [cX, cY]

    # print center coordinates
    #print('center', center)
    #print('currPoint', currPoint)

    # Calculate Euclidean distance
    distance = math.dist(center, currPoint)
    distance = str(round(distance, 2))
    #print('distance', distance)

    cv2.circle(original_image, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(original_image, "centroid", (cX-25, cY-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.circle(original_image, (centerX, centerY), 5, (255, 255, 255), -1)
    cv2.putText(original_image, "centroid", (centerX - 25, centerY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # display the image
    cv2.imshow('original_image', original_image)
    #cv2.imshow('mask', mask)
    #cv2.imshow('thresh', thresh)
    #cv2.waitKey(20000)  # images stay for 20 seconds

    deltaX = cX - centerX
    deltaY = cY - centerY

    # use arctan to get angle in degree
    #degrees_temp = math.atan2(deltaX, deltaY) / math.pi * 180

    #if degrees_temp < 0:
        #degrees_final = 360 + degrees_temp
    #else:
        #degrees_final = degrees_temp

    # We include North twice to counter it being on either side of 0
    #directions = ["North", "North East", "East", "South East", "South", "South West", "West", "North West", "North"]
    # We create a 'score' that will fit our degree value into one of those directions
    # Each bracket is 45 degrees, hence dividing by 45
    #direction_lookup = round(degrees_final / 45)
    # Now, if we look up our value in our directions list, it should return us our direction
    #final_direction = directions[direction_lookup]
    #degrees_final = str(round(degrees_final, 2))
    print("The robot should move horizontally (DeltaX): " + str(deltaX) + " and vertically (DeltaY): "
          + str(deltaY) + " for the full plate view.")

    # Press the green button in the gutter to run the script in PyCharm.

if __name__ == '__main__':
    get_offset_from_partial_plate_view()