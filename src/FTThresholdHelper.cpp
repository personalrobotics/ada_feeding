#include "feeding/FTThresholdHelper.hpp"

#include <thread>

#include <libada/util.hpp>

using ada::util::getRosParam;

namespace feeding {

//==============================================================================
FTThresholdHelper::FTThresholdHelper(bool useThresholdControl,
                                     ros::NodeHandle nodeHandle)
    : mUseThresholdControl(useThresholdControl), mNodeHandle(nodeHandle) {
  mCollectingData.store(false);
  if (!mUseThresholdControl)
    return;
}

//==============================================================================
void FTThresholdHelper::swapTopic(const std::string &topic) {
#ifdef REWD_CONTROLLERS_FOUND
  std::string ftThresholdTopic = topic;
  if (topic == "") {
    ftThresholdTopic = getRosParam<std::string>(
        "ftSensor/controllerFTThresholdTopic", mNodeHandle);
  }
  mFTThresholdClient.reset(
      new rewd_controllers::FTThresholdClient(ftThresholdTopic));
#else
  mUseThresholdControl = false;
#endif
}

//==============================================================================
void FTThresholdHelper::init(bool retare, const std::string &topicOverride) {
  if (!mUseThresholdControl)
    return;

  swapTopic(topicOverride);

#ifdef REWD_CONTROLLERS_FOUND
  auto thresholdPair = getThresholdValues("standard");
  mFTThresholdClient->setThresholds(thresholdPair.first, thresholdPair.second,
                                    retare);
  ROS_WARN_STREAM("initial threshold set finished");

  std::string ftTopic =
      getRosParam<std::string>("ftSensor/ftTopic", mNodeHandle);
  ROS_INFO_STREAM("FTThresholdHelper is listening for " << ftTopic);
  mForceTorqueDataSub = mNodeHandle.subscribe(
      ftTopic, 1, &FTThresholdHelper::forceTorqueDataCallback, this);
#endif
}

//=============================================================================
void FTThresholdHelper::forceTorqueDataCallback(
    const geometry_msgs::WrenchStamped &msg) {
  std::lock_guard<std::mutex> lock(mDataCollectionMutex);
  if (!mCollectingData.load()) {
    return;
  }
  if (mDataPointsToCollect && mCollectedForces.size() >= mDataPointsToCollect) {
    mCollectingData.store(false);
    return;
  }
  Eigen::Vector3d force;
  Eigen::Vector3d torque;
  force.x() = msg.wrench.force.x;
  force.y() = msg.wrench.force.y;
  force.z() = msg.wrench.force.z;
  torque.x() = msg.wrench.torque.x;
  torque.y() = msg.wrench.torque.y;
  torque.z() = msg.wrench.torque.z;
  mCollectedForces.push_back(force);
  mCollectedTorques.push_back(torque);
  mTimestamps.push_back(msg.header.stamp);
}

bool FTThresholdHelper::startDataCollection(size_t numberOfDataPoints) {
  std::lock_guard<std::mutex> lock(mDataCollectionMutex);
  if (mCollectingData.load()) {
    return false;
  }

  mDataPointsToCollect = numberOfDataPoints;
  mCollectedForces.clear();
  mCollectedTorques.clear();
  mTimestamps.clear();
  mCollectingData.store(true);
  return true;
}

void FTThresholdHelper::stopDataCollection() { mCollectingData.store(false); }

bool FTThresholdHelper::isDataCollectionFinished() {
  return !mCollectingData.load();
}

bool FTThresholdHelper::getDataAverage(Eigen::Vector3d &forceMean,
                                       Eigen::Vector3d &torqueMean) {
  std::lock_guard<std::mutex> lock(mDataCollectionMutex);
  if (mCollectingData.load()) {
    return false;
  }

  forceMean.fill(0);
  torqueMean.fill(0);

  if (mCollectedForces.size() == 0) {
    return true;
  }

  Eigen::Vector3d summedForces, summedTorques;
  summedForces.fill(0);
  summedTorques.fill(0);
  for (size_t i = 0; i < mCollectedForces.size(); i++) {
    summedForces = summedForces + mCollectedForces[i];
    summedTorques = summedTorques + mCollectedTorques[i];
  }
  forceMean.x() = summedForces.x() / mCollectedForces.size();
  forceMean.y() = summedForces.y() / mCollectedForces.size();
  forceMean.z() = summedForces.z() / mCollectedForces.size();
  torqueMean.x() = summedTorques.x() / mCollectedTorques.size();
  torqueMean.y() = summedTorques.y() / mCollectedTorques.size();
  torqueMean.z() = summedTorques.z() / mCollectedTorques.size();
  return true;
}

bool FTThresholdHelper::writeDataToFile(const std::string &fileName) {
  if (mCollectingData.load()) {
    return false;
  }

  std::ofstream file(fileName);
  if (!file.is_open()) {
    ROS_WARN_STREAM("FTThresholdHelper: Error opening file");
    return false;
  }

  // Header
  file << "Time (ms), Fx, Fy, Fz, Tx, Ty, Tz" << std::endl;

  for (size_t i = 0; i < mCollectedForces.size(); i++) {
    file << (mTimestamps[i].toNSec() / 1000000) << ", ";
    file << mCollectedForces[i].x() << ", " << mCollectedForces[i].y() << ", "
         << mCollectedForces[i].z() << ", ";
    file << mCollectedTorques[i].x() << ", " << mCollectedTorques[i].y() << ", "
         << mCollectedTorques[i].z() << std::endl;
  }

  file.close();
  return true;
}

std::vector<double> FTThresholdHelper::getData() {
  std::vector<double> ret;

  if (mCollectingData.load()) {
    return ret;
  }

  for (size_t i = 0; i < mCollectedForces.size(); i++) {
    ret.push_back(mTimestamps[i].toNSec() / 1000000);
    ret.push_back(mCollectedForces[i].x());
    ret.push_back(mCollectedForces[i].y());
    ret.push_back(mCollectedForces[i].z());
    ret.push_back(mCollectedTorques[i].x());
    ret.push_back(mCollectedTorques[i].y());
    ret.push_back(mCollectedTorques[i].z());
  }

  return ret;
}

//==============================================================================
bool FTThresholdHelper::setThresholds(std::string preset, bool retare) {
  if (!mUseThresholdControl)
    return true;

#ifdef REWD_CONTROLLERS_FOUND
  auto thresholdPair = getThresholdValues(preset);
  ROS_INFO_STREAM("Set thresholds " << thresholdPair.first << " "
                                    << thresholdPair.second);
  return mFTThresholdClient->setThresholds(thresholdPair.first,
                                           thresholdPair.second, retare);
#endif

  // Handle no rewd_controllers case as if thresholds disabled
  return true;
}

//==============================================================================
bool FTThresholdHelper::setThresholds(double forces, double torques,
                                      bool retare) {
  if (!mUseThresholdControl)
    return true;

#ifdef REWD_CONTROLLERS_FOUND
  ROS_INFO_STREAM("Set thresholds " << forces << " " << torques);
  return mFTThresholdClient->setThresholds(forces, torques, retare);
#endif

  // Handle no rewd_controllers case as if thresholds disabled
  return true;
}

//==============================================================================
std::pair<double, double>
FTThresholdHelper::getThresholdValues(std::string preset) {
  double forceThreshold = 0;
  double torqueThreshold = 0;
  forceThreshold = getRosParam<double>(
      "ftSensor/thresholds/" + preset + "/force", mNodeHandle);
  torqueThreshold = getRosParam<double>(
      "ftSensor/thresholds/" + preset + "/torque", mNodeHandle);
  return std::pair<double, double>(forceThreshold, torqueThreshold);
}
} // namespace feeding
