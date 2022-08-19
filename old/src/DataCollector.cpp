#include "feeding/DataCollector.hpp"

#include <iostream>
#include <sstream>

#include <boost/date_time.hpp>
#include <boost/filesystem/path.hpp>
#include <stdlib.h>
#include <yaml-cpp/yaml.h>

#include <libada/util.hpp>

#include "boost/date_time/posix_time/posix_time.hpp"

using ada::util::createBwMatrixForTSR;
using ada::util::createIsometry;
using ada::util::getRosParam;
using ada::util::waitForUser;
using aikido::constraint::dart::TSR;

namespace {

// Robot To World
static const Eigen::Isometry3d robotPose =
    createIsometry(0.7, -0.1, -0.25, 0, 0, 3.1415);

// Modification of cameraCalibaration's util::getCalibrationTSR
TSR getSideViewTSR(int step) {
  double angle = 0.1745 * step;
  auto tsr = pr_tsr::getDefaultPlateTSR();
  tsr.mT0_w = robotPose.inverse() *
              createIsometry(0.425 + sin(angle) * 0.1 + cos(angle) * -0.03,
                             0.15 - cos(angle) * 0.1 + sin(angle) * -0.03, 0.05,
                             3.58, 0, angle);

  tsr.mBw = createBwMatrixForTSR(0.001, 0.001, 0, 0);
  return tsr;
}
} // namespace
namespace feeding {

//==============================================================================
void createDirectory(const std::string &directory) {

  if (!boost::filesystem::is_directory(directory)) {
    boost::filesystem::create_directories(directory);

    if (!boost::filesystem::is_directory(directory)) {
      ROS_ERROR_STREAM("Could not create " << directory << std::endl);
    } else {
      ROS_INFO_STREAM("Created " << directory << std::endl);
    }
  }
}

//==============================================================================
void setupDirectoryPerData(const std::string &root) {
  std::vector<std::string> folders{"color", "depth"};
  for (auto &folder : folders) {
    auto directory = root + "CameraInfoMsgs/" + folder + "/";
    createDirectory(directory);

    directory = root + folder;
    createDirectory(directory);
  }
  createDirectory(root + "success");
}

//==============================================================================
void removeDirectory(const std::string directory) {
  if (!boost::filesystem::remove_all(directory)) {
    ROS_WARN_STREAM("Failed to remove " << directory);
  } else {
    ROS_WARN_STREAM("Removed " << directory);
  }
}

//==============================================================================
void DataCollector::infoCallback(const sensor_msgs::CameraInfoConstPtr &msg,
                                 ImageType imageType) {
  if (imageType == COLOR && !mShouldRecordColorInfo.load())
    return;
  if (imageType == DEPTH && !mShouldRecordDepthInfo.load())
    return;

  std::string folder = imageType == COLOR ? "color" : "depth";

  if (mShouldRecordColorInfo.load() || mShouldRecordColorInfo.load()) {
    ROS_INFO("recording camera info!");

    auto directory = mDataCollectionPath + "CameraInfoMsgs/" + folder + "/";

    std::string infoFile = directory + "cameraInfo.yaml";

    YAML::Node node;
    node["width"] = msg->width;
    node["height"] = msg->height;

    for (std::size_t i = 0; i < 9; i++)
      node["K"].push_back(*(msg->K.data() + i));

    std::ofstream outFile(infoFile);
    outFile << node;

    ROS_INFO_STREAM("Wrote to " << infoFile);

    if (imageType == COLOR) {
      mShouldRecordColorInfo.store(false);
      std::cout << "color Set to " << mShouldRecordColorInfo.load()
                << std::endl;
    } else {
      mShouldRecordDepthInfo.store(false);
      std::cout << "depth Set to " << mShouldRecordDepthInfo.load()
                << std::endl;
    }
  }
}

//==============================================================================
void DataCollector::imageCallback(const sensor_msgs::ImageConstPtr &msg,
                                  ImageType imageType) {
  if (imageType == COLOR && !mShouldRecordColorImage.load())
    return;
  if (imageType == DEPTH && !mShouldRecordDepthImage.load())
    return;

  std::string folder = imageType == COLOR ? "color" : "depth";
  std::lock_guard<std::mutex> lock(mCallbackLock);

  if (mShouldRecordColorImage.load() || mShouldRecordDepthImage.load()) {

    if (imageType == COLOR) {
      mShouldRecordColorImage.store(false);
    } else {
      mShouldRecordDepthImage.store(false);
    }
    ROS_INFO("recording image!");

    cv_bridge::CvImagePtr cv_ptr;
    try {
      if (imageType == ImageType::COLOR) {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      } else {
        cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
      }
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    auto count =
        imageType == COLOR ? mColorImageCount.load() : mDepthImageCount.load();

    std::string imageFile = mDataCollectionPath + folder + +"/image_" +
                            std::to_string(count) + ".png";
    bool worked = cv::imwrite(imageFile, cv_ptr->image);
    std::cout << "Trying to save at " << imageFile << std::endl;

    if (imageType == COLOR) {
      mColorImageCount++;
      mShouldRecordColorImage.store(false);
    } else {
      mDepthImageCount++;
      mShouldRecordDepthImage.store(false);
    }

    if (worked)
      ROS_INFO_STREAM("image saved to " << imageFile);
    else
      ROS_WARN_STREAM("image saving failed");
  }
}

//==============================================================================
DataCollector::DataCollector(std::shared_ptr<FeedingDemo> feedingDemo,
                             std::shared_ptr<ada::Ada> ada,
                             ros::NodeHandle nodeHandle, bool autoContinueDemo,
                             bool adaReal, bool perceptionReal,
                             const std::string &dataCollectionPath)
    : mFeedingDemo(std::move(feedingDemo)), mAda(std::move(ada)),
      mNodeHandle(nodeHandle), mAutoContinueDemo(autoContinueDemo),
      mAdaReal(adaReal), mDataCollectionPath{dataCollectionPath},
      mPerceptionReal{perceptionReal}, mShouldRecordColorImage{false},
      mShouldRecordDepthImage{false}, mShouldRecordColorInfo{false},
      mShouldRecordDepthInfo{false}, mCurrentFood{0}, mCurrentDirection{0},
      mCurrentTrial{0}, mColorImageCount{0}, mDepthImageCount{0} {
  // See if we can save force/torque sensor data as well.

  // Set Standard Threshold
  mFeedingDemo->setFTThreshold(STANDARD_FT_THRESHOLD);

  mNumTrials = getRosParam<int>("/data/numTrials", mNodeHandle);
  mFoods = getRosParam<std::vector<std::string>>("/data/foods", mNodeHandle);
  mTiltAngles =
      getRosParam<std::vector<double>>("/data/tiltAngles", mNodeHandle);
  mTiltModes = getRosParam<std::vector<int>>("/data/tiltModes", mNodeHandle);
  mDirections =
      getRosParam<std::vector<double>>("/data/directions", mNodeHandle);
  mAngleNames =
      getRosParam<std::vector<std::string>>("/data/angleNames", mNodeHandle);

  if (mAdaReal || mPerceptionReal) {
    image_transport::ImageTransport it(mNodeHandle);
    sub = it.subscribe(
        "/camera/color/image_raw", 1,
        boost::bind(&DataCollector::imageCallback, this, _1, ImageType::COLOR));
    sub2 = it.subscribe(
        "/camera/aligned_depth_to_color/image_raw", 1,
        boost::bind(&DataCollector::imageCallback, this, _1, ImageType::DEPTH));
    sub3 = mNodeHandle.subscribe<sensor_msgs::CameraInfo>(
        "/camera/color/camera_info", 1,
        boost::bind(&DataCollector::infoCallback, this, _1, ImageType::COLOR));
    sub4 = mNodeHandle.subscribe<sensor_msgs::CameraInfo>(
        "/camera/aligned_depth_to_color/camera_info", 1,
        boost::bind(&DataCollector::infoCallback, this, _1, ImageType::DEPTH));
  }

  mPlanningTimeout =
      getRosParam<double>("/planning/timeoutSeconds", mNodeHandle);
  mMaxNumPlanningTrials =
      getRosParam<int>("/planning/maxNumberOfTrials", mNodeHandle);

  mEndEffectorOffsetPositionTolerance = getRosParam<double>(
      "/planning/endEffectorOffset/positionTolerance", mNodeHandle),
  mEndEffectorOffsetAngularTolerance = getRosParam<double>(
      "/planning/endEffectorOffset/angularTolerance", mNodeHandle);
}

//==============================================================================
void DataCollector::setDataCollectionParams(int foodId, int pushDirectionId,
                                            int trialId) {
  if (mAdaReal) {
    std::lock_guard<std::mutex> lockImage(mCallbackLock);

    std::this_thread::sleep_for(std::chrono::milliseconds(3));

    // Update only when positive.
    if (foodId != -1) {
      mCurrentFood.store(foodId);
      mCurrentDirection.store(pushDirectionId);
      mCurrentTrial.store(trialId);
    }

    mShouldRecordColorInfo.store(true);
    mShouldRecordDepthInfo.store(true);
  }
  // wait for first stream to be saved
  /*
  while (mShouldRecordColorInfo.load() || mShouldRecordDepthInfo.load())
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::cout << "Data collector complete" << std::endl;
  */
}

//==============================================================================
void DataCollector::collect(Action action, const std::string &foodName,
                            std::size_t directionIndex,
                            std::size_t trialIndex) {
  if (directionIndex >= mDirections.size()) {
    std::stringstream ss;
    ss << "Direction index [" << directionIndex
       << "] is greater than the max index [" << mDirections.size() - 1
       << "].\n";
    throw std::invalid_argument(ss.str());
  }
  if (trialIndex >= mNumTrials) {
    std::stringstream ss;
    ss << "Trial index [" << trialIndex << "] is greater than the max index ["
       << mNumTrials - 1 << "].\n";
    throw std::invalid_argument(ss.str());
  }
  if (std::find(mFoods.begin(), mFoods.end(), foodName) == mFoods.end()) {
    std::stringstream ss;
    ss << "Food " << foodName << " not in the list of foods.\n";
    throw std::invalid_argument(ss.str());
  }

  std::string trialName = ActionToString.at(action) + "/" + foodName +
                          "-angle-" + mAngleNames[directionIndex] + "-trial-" +
                          std::to_string(trialIndex);
  mDataCollectionPath += trialName + "/";
  setupDirectoryPerData(mDataCollectionPath);

  auto foodIndex = std::distance(
      mFoods.begin(), std::find(mFoods.begin(), mFoods.end(), foodName));

  ROS_INFO_STREAM("\nTrial " << trialIndex << ": Food [" << foodName
                             << "] Direction [" << mAngleNames[directionIndex]
                             << "] \n\n");

  std::cout << "Set data collection params." << std::endl;
  ;

  float rotateForqueAngle = mDirections[directionIndex] * M_PI / 180.0;

  setDataCollectionParams(foodIndex, directionIndex, trialIndex);

  ROS_INFO("Starting data collection");

  bool result;
  if (action == VERTICAL_SKEWER) {
    result = skewer(rotateForqueAngle, TiltStyle::NONE);
  } else if (action == TILTED_VERTICAL_SKEWER) {
    std::stringstream ss;
    ss << "Rotate the food " << mDirections[directionIndex] << " degrees"
       << std::endl;
    mFeedingDemo->waitForUser(ss.str());

    if (!skewer(0, TiltStyle::VERTICAL)) {
      ROS_INFO_STREAM("Terminating.");
      return;
    }
  } else if (action == TILTED_ANGLED_SKEWER) {
    std::stringstream ss;
    ss << "Rotate the food " << rotateForqueAngle << " degrees" << std::endl;
    mFeedingDemo->waitForUser(ss.str());

    if (!skewer(0, TiltStyle::ANGLED)) {
      ROS_INFO_STREAM("Terminating.");
      return;
    }
  } else if (action == SCOOP) {
    mFeedingDemo->scoop();
  }

  if (!result) {
    ROS_INFO_STREAM("Terminating.");
    return;
  }

  recordSuccess();

  ROS_INFO_STREAM("Terminating.");
  return;
}

//==============================================================================
void DataCollector::collect_images(const std::string &foodName) {
  ROS_INFO_STREAM("Collect images for " << foodName);

  mDataCollectionPath = mDataCollectionPath + "/" + foodName + "/";
  setupDirectoryPerData(mDataCollectionPath);
  setDataCollectionParams(0, 0, 0);

  ROS_INFO_STREAM("Update image counts");
  updateImageCounts(mDataCollectionPath, ImageType::COLOR);
  updateImageCounts(mDataCollectionPath, ImageType::DEPTH);

  // Move above food (center of plate)
  ROS_INFO_STREAM("Move above food");
  if (!mFeedingDemo->moveAboveFood("", // Ignore name.
                                   mFeedingDemo->getDefaultFoodTransform(), 0.0,
                                   TiltStyle::NONE)) {
    ROS_ERROR("Rotate Forque failed. Restart.");
    return;
  }

  captureFrame();

  // Move in a few centimeters.
  ROS_INFO_STREAM("Move in 2.5 cm.");
  double length = 0.025;
  if (!mFeedingDemo->getAda()->moveArmToEndEffectorOffset(
          Eigen::Vector3d(0, 0, -1), length, nullptr, mPlanningTimeout,
          mEndEffectorOffsetPositionTolerance,
          mEndEffectorOffsetAngularTolerance)) {
    ROS_ERROR("Rotate Forque failed. Restart.");
    return;
  }

  captureFrame();

  // Rotate around and take photos.
  // Modification of calibration viewpoints.
  ROS_INFO_STREAM("Rotate around.");

  for (int i = 0; i < 100; i += 10) {
    auto tsr = getSideViewTSR(i);

    // auto marker = mFeedingDemo->getViewer()->addTSRMarker(tsr);
    // mFeedingDemo->waitForUser("Check");

    if (!mFeedingDemo->getAda()->moveArmToTSR(
            tsr, mFeedingDemo->getCollisionConstraint(), mPlanningTimeout,
            mMaxNumPlanningTrials)) {
      ROS_INFO_STREAM("Fail: Step " << i);
    } else {
      captureFrame();
    }
  }
  return;
}

//==============================================================================
bool DataCollector::skewer(float rotateForqueAngle, TiltStyle tiltStyle) {
  // ===== ROTATE FORQUE ====
  if (!mFeedingDemo->moveAboveFood("", // Ignore name.
                                   mFeedingDemo->getDefaultFoodTransform(),
                                   rotateForqueAngle, tiltStyle)) {
    ROS_ERROR("Rotate Forque failed. Restart.");
    removeDirectory(mDataCollectionPath);
    return false;
  }
  captureFrame();

  // ===== INTO TO FOOD ====
  Eigen::Vector3d direction(0, 0, -1);
  if (tiltStyle == TiltStyle::ANGLED) {
    Eigen::Vector3d food(mFeedingDemo->getDefaultFoodTransform().translation());
    Eigen::Vector3d hand(mFeedingDemo->getAda()
                             ->getHand()
                             ->getEndEffectorBodyNode()
                             ->getTransform()
                             .translation());
    direction = food - hand;
    direction.normalize();
  }

  mFeedingDemo->moveInto(TargetItem::FOOD, tiltStyle, direction);
  captureFrame();

  // ===== OUT OF FOOD =====
  mFeedingDemo->setFTThreshold(AFTER_GRAB_FOOD_FT_THRESHOLD);
  mFeedingDemo->moveOutOf(TargetItem::FOOD, true);
  captureFrame();

  return true;
}

//==============================================================================
void DataCollector::recordSuccess() {
  std::string food = mFoods[mCurrentFood.load()];
  std::string direction = mAngleNames[mCurrentDirection.load()];
  int trial = mCurrentTrial.load();

  auto fileName = mDataCollectionPath + "success/" + food + "-" + direction +
                  "-" + std::to_string(trial) + ".txt";

  ROS_INFO_STREAM("Record success for " << food << " direction " << direction
                                        << " trial " << trial << " [y/n]");

  std::vector<std::string> optionPrompts{"(1) success", "(2) fail",
                                         "(3) delete"};
  auto input = getUserInputWithOptions(optionPrompts, "Did I succeed?");
  if (input == 1) {
    ROS_INFO("Recording success");
    std::ofstream ss;
    ss.open(fileName);
    ROS_INFO_STREAM(fileName);
    ss << "success" << std::endl;
    ss.close();
  } else if (input == 2) {
    ROS_INFO("Recording failure");
    std::ofstream ss;
    ss.open(fileName);
    ROS_INFO_STREAM(fileName);
    ss << "fail" << std::endl;
    ss.close();
  } else {
    ROS_ERROR_STREAM("Removing data " << mDataCollectionPath);
    removeDirectory(mDataCollectionPath);
  }
}

//==============================================================================
void DataCollector::captureFrame() {
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  mCallbackLock.lock();
  mShouldRecordColorImage.store(true);
  mShouldRecordDepthImage.store(true);
  mCallbackLock.unlock();
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}
//==============================================================================
void DataCollector::updateImageCounts(const std::string &directory,
                                      ImageType imageType) {
  using namespace boost::filesystem;

  std::string folder = directory + (imageType == COLOR ? "color" : "depth");
  int count = 0;
  for (directory_iterator it(directory); it != directory_iterator(); ++it) {
    if (is_regular_file(it->status()))
      count++;
  }

  if (imageType == COLOR) {
    mColorImageCount.store(count);
  } else {
    mDepthImageCount.store(count);
  }
  ROS_INFO_STREAM(folder + " has " << count << "images");
}

} // namespace feeding
