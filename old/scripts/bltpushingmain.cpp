#include <aikido/rviz/InteractiveMarkerViewer.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pr_tsr/plate.hpp>
#include <ros/ros.h>
#include "feeding/FTThresholdHelper.hpp"
#include "feeding/FeedingDemo.hpp"
#include "feeding/Perception.hpp"
#include "feeding/util.hpp"

namespace feeding {

std::atomic<bool> shouldRecordImagePush{false};

std::string return_current_time_and_date_push()
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
  return ss.str();
}

void imageCallbackPush(const sensor_msgs::ImageConstPtr& msg)
{

  if (shouldRecordImagePush.load())
  {
    ROS_ERROR("recording image!");

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // cv::namedWindow("image", WINDOW_AUTOSIZE);
    // cv::imshow("image", cv_ptr->image);
    // cv::waitKey(30);

    static int image_count = 0;
    // std::stringstream sstream;
    std::string imageFile
        = "/home/herb/Images/" + return_current_time_and_date_push() + ".png";
    // sstream << imageFile;
    bool worked = cv::imwrite(imageFile, cv_ptr->image);
    image_count++;
    ROS_INFO_STREAM("image saved to " << imageFile << ", worked: " << worked);
    shouldRecordImagePush.store(false);
  }
}

int bltpushingmain(
    FeedingDemo& feedingDemo,
    FTThresholdHelper& ftThresholdHelper,
    Perception& perception,
    aikido::rviz::InteractiveMarkerViewerPtr viewer,
    ros::NodeHandle nodeHandle,
    bool autoContinueDemo,
    bool adaReal)
{

  // Set Standard Threshold
  if (!ftThresholdHelper.setThresholds(STANDARD_FT_THRESHOLD))
  {
    return 1;
  }

  image_transport::ImageTransport it(nodeHandle);
  // ros::Subscriber sub_info =
  // nodeHandle.subscribe("/camera/color/camera_info", 1, cameraInfo);
  image_transport::Subscriber sub = it.subscribe(
      "/data_collection/target_image",
      1,
      imageCallbackPush /*, image_transport::TransportHints("compressed")*/);

  int numTrials = getRosParam<int>("/numTrials", nodeHandle);
  for (int trial = 0; trial < numTrials; trial++)
  {
    // ===== ABOVE PLATE =====
    if (!autoContinueDemo)
    {
      if (!waitForUser("Move forque above plate"))
      {
        return 0;
      }
    }
    feedingDemo.moveAbovePlate();

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    shouldRecordImagePush.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    while (shouldRecordImagePush.load())
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // ===== ABOVE FOOD =====
    std::vector<std::string> foodNames
        = getRosParam<std::vector<std::string>>("/foodItems/names", nodeHandle);
    std::vector<double> skeweringForces
        = getRosParam<std::vector<double>>("/foodItems/forces", nodeHandle);
    std::unordered_map<std::string, double> foodSkeweringForces;
    for (int i = 0; i < foodNames.size(); i++)
    {
      foodSkeweringForces[foodNames[i]] = skeweringForces[i];
    }

    Eigen::Isometry3d foodTransform;
    bool foodFound = false;
    std::string foodName;
    while (!foodFound)
    {
      std::cout << std::endl
                << "\033[1;32mWhich food item do you want?\033[0m     > ";
      foodName = "";
      std::cin >> foodName;
      if (!ros::ok())
      {
        return 0;
      }
      if (!perception.setFoodName(foodName))
      {
        std::cout << "\033[1;33mI don't know about any food that's called '"
                  << foodName << ". Wanna get something else?\033[0m"
                  << std::endl;
        continue;
      }

      if (adaReal)
      {
        bool perceptionSuccessful
            = perception.perceiveFood(foodTransform, true, viewer);
        if (!perceptionSuccessful)
        {
          std::cout << "\033[1;33mI can't see the " << foodName
                    << "... Wanna get something else?\033[0m" << std::endl;
          continue;
        }
        else
        {
          foodFound = true;
        }
      }
      else
      {
        foodTransform = feedingDemo.getDefaultFoodTransform();
        foodFound = true;
      }
    }
    std::cout << "\033[1;32mAlright! Let's get the " << foodName
              << "!\033[0;32m  (Gonna skewer with "
              << foodSkeweringForces[foodName] << "N)\033[0m" << std::endl
              << std::endl;

    bool foodPickedUp = false;
    while (!foodPickedUp)
    {

      if (!autoContinueDemo)
      {
        if (!waitForUser("Move forque above food"))
        {
          return 0;
        }
      }
      feedingDemo.moveAboveFood(foodTransform, 0, viewer);

      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      shouldRecordImagePush.store(true);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      while (shouldRecordImagePush.load())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      if (adaReal)
      {
        bool perceptionSuccessful
            = perception.perceiveFood(foodTransform, true, viewer);
        if (!perceptionSuccessful)
        {
          std::cout << "\033[1;33mI can't see the " << foodName
                    << " anymore...\033[0m" << std::endl;
        }
        else
        {
          feedingDemo.moveAboveFood(foodTransform, 0, viewer);

          std::this_thread::sleep_for(std::chrono::milliseconds(2000));
          shouldRecordImagePush.store(true);
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          while (shouldRecordImagePush.load())
          {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        }
      }

      double zForceBeforeSkewering = 0;
      if (adaReal && ftThresholdHelper.startDataCollection(20))
      {
        Eigen::Vector3d currentForce, currentTorque;
        while (!ftThresholdHelper.isDataCollectionFinished(
            currentForce, currentTorque))
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        zForceBeforeSkewering = currentForce.z();
      }

      // ===== ROTATE FORQUE ====
      std::cout << std::endl
                << "\033[1;32mWhat angle do you want to push food at in "
                   "degrees?\033[0m     > ";
      float angle = 0;
      std::cin >> angle;
      angle *= M_PI / 180.0;
      if (!ros::ok())
      {
        return 0;
      }

      if (!autoContinueDemo)
      {
        if (!waitForUser("Rotate forque in orientation to push food"))
        {
          return 0;
        }
      }
      feedingDemo.rotateForque(foodTransform, angle, viewer);

      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      shouldRecordImagePush.store(true);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      while (shouldRecordImagePush.load())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      /*if (adaReal) {
          feedingDemo.moveNextToFood(&perception, angle, viewer);
      } else {
          feedingDemo.moveNextToFood(foodTransform, angle, viewer);
      }*/

      // ===== NEXT TO FOOD ====
      if (!autoContinueDemo)
      {
        if (!waitForUser("Move forque next to food"))
        {
          return 0;
        }
      }
      double torqueThreshold = 2;
      if (!ftThresholdHelper.setThresholds(
              foodSkeweringForces[foodName], torqueThreshold))
      {
        return 1;
      }
      Eigen::Isometry3d forqueTransform;
      if (adaReal)
      {
        forqueTransform = perception.getForqueTransform();
      }
      if (adaReal)
      {
        feedingDemo.moveNextToFood(&perception, angle, viewer, forqueTransform);
      }
      else
      {
        feedingDemo.moveNextToFood(foodTransform, angle, viewer);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      shouldRecordImagePush.store(true);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      while (shouldRecordImagePush.load())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      // ===== MOVE OUT OF PLATE ====
      if (!autoContinueDemo)
      {
        if (!waitForUser("Move Out of Plate"))
        {
          return 0;
        }
      }
      if (!ftThresholdHelper.setThresholds(AFTER_GRAB_FOOD_FT_THRESHOLD))
      {
        return 1;
      }
      feedingDemo.moveOutOfPlate();

      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      shouldRecordImagePush.store(true);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      while (shouldRecordImagePush.load())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      // ===== PUSH FOOD ====
      if (!autoContinueDemo)
      {
        if (!waitForUser("Push food"))
        {
          return 0;
        }
      }
      if (!ftThresholdHelper.setThresholds(
              foodSkeweringForces[foodName], torqueThreshold))
      {
        return 1;
      }
      // feedingDemo.grabFoodWithForque();

      if (!ftThresholdHelper.setThresholds(PUSH_FOOD_FT_THRESHOLD))
      {
        return 1;
      }

      if (adaReal)
      {
        feedingDemo.pushFood(angle, forqueTransform);
      }
      else
      {
        feedingDemo.pushFood(angle);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      shouldRecordImagePush.store(true);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      while (shouldRecordImagePush.load())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      break;
    }
    // ===== OUT OF FOOD =====
    if (!autoContinueDemo)
    {
      if (!waitForUser("Move forque out of food"))
      {
        return 0;
      }
    }
    if (!ftThresholdHelper.setThresholds(AFTER_GRAB_FOOD_FT_THRESHOLD))
    {
      return 1;
    }
    feedingDemo.moveOutOfFood();

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    shouldRecordImagePush.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    while (shouldRecordImagePush.load())
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    /*    std::string doneResponse;
        std::cout << std::endl << "\033[1;32mShould we keep going? [y/n]\033[0m
       > ";
        doneResponse = "";
        std::cin >> doneResponse;
        if (!ros::ok()) {return 0;}
        if (doneResponse == "n") {
          done = true;
        }*/
  }

  // ===== DONE =====
  waitForUser("Demo finished.");
}
};
