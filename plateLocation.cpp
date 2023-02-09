#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;
using namespace cv;

int main() {
    // load an image using 'imread'
    Mat original_image = imread("/Users/raidakarim/Downloads/half_blue_plate.png");
    int h = original_image.rows;
    int w = original_image.cols;
    // where w / 2, h / 2 are the required frame/image centeroid's XYcoordinates.
    int centerX = w / 2;
    int centerY = h / 2;
    cout << "centerX: " << centerX << endl;
    cout << "centerY: " << centerY << endl;
    // original_image = cv2.imread("/Users/raidakarim/Downloads/full_blue_plate.png")
    // convert images from BGR (Blue, Green, Red) to HSV (Hue-- color, Saturation-- density, Value-- lightness)
    Mat hsv_image;
    cvtColor(original_image, hsv_image, COLOR_BGR2HSV);
    /// blue color detection and masking ///
    Scalar lower_blue = Scalar(94, 80, 2);
    Scalar upper_blue = Scalar(126, 255, 255);
    // define range of blue color in HSV here to create a mask of blue colored object
    Mat mask;
    inRange(hsv_image, lower_blue, upper_blue, mask);

    // convert the mask image to binary image
    // The method returns two outputs.
    // The first is the threshold that was used and the second output is the thresholded image.
    // https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    // ret ,thresh = cv::threshold(mask, 127, 255, 0);

    // calculate moments of binary image
    Moments m = moments(mask, true);
    cout << m << endl;

    // calculate x,y coordinates of center
    int cX = int(m.m10 / m.m00);
    int cY = int(m.m01 / m.m00);

    Point center = Point(centerX, centerY);
    Point currPoint = Point(cX, cY);

    // print center coordinates
    cout << "center: " << center << endl;
    cout << "currPoint: " << currPoint << endl;

    // Calculate Euclidean distance
    double distance = norm(center - currPoint);
    distance = round(distance, 2);
    cout << "distance: " << distance << endl;

    circle(original_image, currPoint, 5, Scalar(255, 255, 255), -1);
    putText(original_image, "centroid", Point(cX - 25, cY - 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
    circle(original_image, center, 5, Scalar(255, 255, 255), -1);
    putText(original_image, "centroid", Point(centerX - 25, centerY - 25), FONT_HERSHEY_SIMPLEX, 0.5,
            Scalar(255, 255, 255), 2);
    // display the image
    // cv::imshow("original_image", original_image);
    // cv::imshow("mask", mask);
    // cv::imshow("thresh", thresh);
    // cv::waitKey(20000);  # images stay for 20 seconds

    int deltaX = centerX - cX;
    int deltaY = centerY - cY;

    // use arctan to get angle in degree
    double degrees_temp = atan2(deltaX, deltaY) / M_PI * 180;

    double degrees_final;
    if (degrees_temp < 0) {
        degrees_final = 360 + degrees_temp;
    } else {
        degrees_final = degrees_temp;
    }

    // We include North twice to counter it being on either side of 0
    vector<string> directions = {"North", "North East", "East", "South East", "South", "South West", "West", "North West", "North"};
    // We create a 'score' that will fit our degree value into one of those directions
    // Each bracket is 45 degrees, hence dividing by 45
    int direction_lookup = round(degrees_final / 45);
    // Now, if we look up our value in our directions list, it should return us our direction
    string final_direction = directions[direction_lookup];
    degrees_final = str(round(degrees_final, 2));
    cout << "The robot should move " << str(distance) << " unit "
          << str(final_direction) << " or in " << degrees_final << " degree angle from current location ("
          << str(cX) << ", " << str(cY) << ") to center location ("
          << str(centerX) << ", " << str(centerY) << ") for the full plate view." << endl;

    return 0;
}