#include "feeding/FTThresholdHelper.hpp"

#include <thread>

#include <libada/util.hpp>

using ada::util::getRosParam;

namespace feeding {

//==============================================================================
FTThresholdHelper::FTThresholdHelper(bool useThresholdControl,
                                     ros::NodeHandle nodeHandle)
    : mUseThresholdControl(useThresholdControl), mNodeHandle(nodeHandle) {
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
  if ((int)mCollectedForces.size() >= mDataPointsToCollect) {
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
}

bool FTThresholdHelper::startDataCollection(int numberOfDataPoints) {
  if (!mUseThresholdControl)
    return false;
  std::lock_guard<std::mutex> lock(mDataCollectionMutex);
  mDataPointsToCollect = numberOfDataPoints;
  mCollectedForces.clear();
  mCollectedTorques.clear();
  return true;
}

bool FTThresholdHelper::isDataCollectionFinished(Eigen::Vector3d &forces,
                                                 Eigen::Vector3d &torques) {
  std::lock_guard<std::mutex> lock(mDataCollectionMutex);
  forces.fill(0);
  torques.fill(0);
  if ((int)mCollectedForces.size() < mDataPointsToCollect) {
    return false;
  }
  Eigen::Vector3d summedForces, summedTorques;
  summedForces.fill(0);
  summedTorques.fill(0);
  for (int i = 0; i < (int)mCollectedForces.size(); i++) {
    summedForces = summedForces + mCollectedForces[i];
    summedTorques = summedTorques + mCollectedTorques[i];
  }
  forces.x() = summedForces.x() / mCollectedForces.size();
  forces.y() = summedForces.y() / mCollectedForces.size();
  forces.z() = summedForces.z() / mCollectedForces.size();
  torques.x() = summedTorques.x() / mCollectedForces.size();
  torques.y() = summedTorques.y() / mCollectedForces.size();
  torques.z() = summedTorques.z() / mCollectedForces.size();
  return true;
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
