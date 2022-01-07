#ifndef FEEDING_DATACOLLECTOR_HPP_
#define FEEDING_DATACOLLECTOR_HPP_

#include <fstream>
#include <iostream>

#include <aikido/rviz/InteractiveMarkerViewer.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pr_tsr/plate.hpp>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sensor_msgs/CameraInfo.h>

#include <libada/Ada.hpp>

#include "feeding/FTThresholdHelper.hpp"
#include "feeding/FeedingDemo.hpp"
#include "feeding/perception/Perception.hpp"
#include "feeding/util.hpp"

namespace feeding {

// Action types for data collection
enum Action {
  VERTICAL_SKEWER,
  TILTED_VERTICAL_SKEWER,
  TILTED_ANGLED_SKEWER,
  SCOOP,
  IMAGE_ONLY
};

enum ImageType { COLOR, DEPTH };

static const std::map<const std::string, Action> StringToAction{
    {"collect_skewer", VERTICAL_SKEWER},
    {"collect_tilted_vertical_skewer", TILTED_VERTICAL_SKEWER},
    {"collect_tilted_angled_skewer", TILTED_ANGLED_SKEWER},
    {"collect_images", IMAGE_ONLY}};

static const std::map<Action, const std::string> ActionToString{
    {VERTICAL_SKEWER, "collect_skewer"},
    {TILTED_VERTICAL_SKEWER, "collect_tilted_vertical_skewer"},
    {TILTED_ANGLED_SKEWER, "collect_tilted_angled_skewer"},
    {IMAGE_ONLY, "collect_images"}};

class DataCollector {
public:
  /// Constructor
  /// \param[in] dataCollectionPath Directory to save all data collection.
  explicit DataCollector(std::shared_ptr<FeedingDemo> feedingDemo,
                         std::shared_ptr<ada::Ada> ada,
                         ros::NodeHandle nodeHandle, bool autoContinueDemo,
                         bool adaReal, bool perceptionReal,
                         const std::string &dataCollectionPath =
                             "/home/herb/feeding/data_collection");

  /// Collect data.
  /// \param[in] action Action to execute
  /// \param[in] foodName Name of food for the data collection
  /// \param[in] directionIndex Index of the direction as suggested by config
  /// file
  /// \param[in] trialIndex Index of trial
  void collect(Action action, const std::string &foodName,
               std::size_t directionIndex, std::size_t trialIndex);

  /// Collect images from multiple views.
  /// Does not perform any bite acquisition actions.
  void collect_images(const std::string &foodName);

private:
  void setDataCollectionParams(int foodId, int pushDirectionId, int trialId);

  void infoCallback(const sensor_msgs::CameraInfoConstPtr &msg,
                    ImageType imageType);

  void imageCallback(const sensor_msgs::ImageConstPtr &msg,
                     ImageType imageType);

  bool skewer(float rotateForqueAngle, TiltStyle tiltStyle);

  void recordSuccess();

  void captureFrame();

  /// Update mColorImageCount and mDepthImageCount to match
  /// the number of images in the respective directories.
  void updateImageCounts(const std::string &directory, ImageType imageType);

  std::shared_ptr<FeedingDemo> mFeedingDemo;
  std::shared_ptr<ada::Ada> mAda;

  ros::NodeHandle mNodeHandle;
  const bool mAutoContinueDemo;
  const bool mAdaReal;
  const bool mPerceptionReal;
  std::string mDataCollectionPath;

  int mNumTrials;
  std::vector<std::string> mFoods;
  std::vector<double> mTiltAngles;
  std::vector<int> mTiltModes;
  std::vector<double> mDirections;
  std::vector<std::string> mAngleNames;

  image_transport::Subscriber sub;
  image_transport::Subscriber sub2;
  ros::Subscriber sub3;
  ros::Subscriber sub4;

  std::atomic<bool> mShouldRecordColorImage;
  std::atomic<bool> mShouldRecordDepthImage;
  std::atomic<bool> mShouldRecordColorInfo;
  std::atomic<bool> mShouldRecordDepthInfo;
  std::atomic<bool> isAfterPush;
  std::atomic<int> mCurrentFood;
  std::atomic<int> mCurrentDirection;
  std::atomic<int> mCurrentTrial;
  std::atomic<int> mColorImageCount;
  std::atomic<int> mDepthImageCount;

  std::mutex mCallbackLock;
  std::mutex mCameraInfoCallbackLock;

  double mPlanningTimeout;
  int mMaxNumPlanningTrials;
  double mEndEffectorOffsetPositionTolerance;
  double mEndEffectorOffsetAngularTolerance;
};

} // namespace feeding

#endif
