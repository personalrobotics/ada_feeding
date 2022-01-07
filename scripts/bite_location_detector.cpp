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
#include "feeding/perception/Perception.hpp"
#include "feeding/util.hpp"

namespace feeding {

std::atomic<bool> shouldRecordImage{false};

std::string return_current_time_and_date()
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
  return ss.str();
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{

  if (shouldRecordImage.load())
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
        = "/home/herb/Images/" + return_current_time_and_date() + ".png";
    // sstream << imageFile;
    bool worked = cv::imwrite(imageFile, cv_ptr->image);
    image_count++;
    ROS_INFO_STREAM("image saved to " << imageFile << ", worked: " << worked);
    shouldRecordImage.store(false);
  }
}

void bite_location_detector(
    FeedingDemo& feedingDemo,
    FTThresholdHelper& ftThresholdHelper,
    Perception& perception,
    ros::NodeHandle nodeHandle,
    bool autoContinueDemo,
    bool adaReal)
{

  aikido::rviz::InteractiveMarkerViewerPtr viewer
      = feedingDemo.getViewer();

  bool collectData = false;

  image_transport::ImageTransport it(nodeHandle);
  // ros::Subscriber sub_info =
  // nodeHandle.subscribe("/camera/color/camera_info", 1, cameraInfo);
  image_transport::Subscriber sub = it.subscribe(
      "/data_collection/target_image",
      1,
      imageCallback /*, image_transport::TransportHints("compressed")*/);

  if (!autoContinueDemo)
  {
    if (!waitForUser("Ready to start."))
    {
      return 0;
    }
  }

  for (int trial = 0; trial < 100; trial++)
  {
    std::cout << "\033[1;33mSTARTING TRIAL " << trial << "\033[0m" << std::endl;

    if (collectData)
    {
      // ===== ABOVE PLATE =====
      bool stepSuccessful = false;
      while (!stepSuccessful)
      {
        try
        {
          feedingDemo.moveAbovePlateAnywhere();
          stepSuccessful = true;
        }
        catch (std::runtime_error)
        {
          if (!waitForUser("Trajectory execution failed. Try again?"))
          {
            continue;
          }
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      shouldRecordImage.store(true);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      while (shouldRecordImage.load())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    else
    {
      feedingDemo.moveAbovePlate();
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      Eigen::Isometry3d foodTransform;
      perception.setFoodName("celery");
      bool perceptionSuccessful = false;
      while (!perceptionSuccessful)
      {
        perceptionSuccessful
            = perception.perceiveFood(foodTransform, true, viewer);
        if (!perceptionSuccessful)
        {
          if (!waitForUser("Perception failed. Try again?"))
          {
            return 0;
          }
        }
      }

      if (!waitForUser("Continue"))
      {
        return 0;
      }

      // ABOVE FOOD
      auto aboveFoodTSR = pr_tsr::getDefaultPlateTSR();
      aboveFoodTSR.mT0_w = foodTransform;
      aboveFoodTSR.mTw_e.translation() = Eigen::Vector3d{0, 0, -0.03};
      aboveFoodTSR.mBw = createBwMatrixForTSR(0.001, 0.001, 0, 0);
      // TODO: remove hardcoded transform for food
      Eigen::Isometry3d eeTransform;
      Eigen::Matrix3d rot;
      rot << -1, 0., 0.,
            0., 1., 0.,
            0., 0., -1;
      eeTransform.linear() = rot;
      aboveFoodTSR.mTw_e.matrix() *= eeTransform.matrix();

      bool trajectoryCompleted
          = feedingDemo.mAdaMover->moveArmToTSR(aboveFoodTSR);
      if (!trajectoryCompleted)
      {
        throw std::runtime_error("Trajectory execution failed");
      }
    }

    // ===== INTO FOOD =====
    // if (!autoContinueDemo)
    // {
    //   if (!waitForUser("Move forque into food"))
    //   {
    //     return 0;
    //   }
    // }
    if (!ftThresholdHelper.setThresholds(25, 0.5))
    {
      return 1;
    }
    // for (int i=0; i<4; i++) {
    //   feedingDemo.mAdaMover->moveToEndEffectorOffset(Eigen::Vector3d(0, 0,
    //   -1), -0.025, false);
    // }
    feedingDemo.mAdaMover->moveToEndEffectorOffset(
        Eigen::Vector3d(0, 0, -1), 0.07, false);

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
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    feedingDemo.mAdaMover->moveToEndEffectorOffset(
        Eigen::Vector3d(0, 0, 1), 0.07, false);
    if (!ftThresholdHelper.setThresholds(STANDARD_FT_THRESHOLD))
    {
      return 1;
    }
  }

  // ===== DONE =====
  waitForUser("Demo finished.");
}
};
