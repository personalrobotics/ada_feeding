#ifndef FEEDING_FTTHRESHOLDHELPER_HPP_
#define FEEDING_FTTHRESHOLDHELPER_HPP_

#include <geometry_msgs/WrenchStamped.h>

#ifdef REWD_CONTROLLERS_FOUND
#include <rewd_controllers/FTThresholdClient.hpp>
#endif
#include <mutex>

#include <Eigen/Geometry>
#include <ros/ros.h>

namespace feeding {

enum FTThreshold {
  STANDARD_FT_THRESHOLD,
  GRAB_FOOD_FT_THRESHOLD,
  AFTER_GRAB_FOOD_FT_THRESHOLD,
  PUSH_FOOD_FT_THRESHOLD
};

/// The FTThresholdHelper configures the MoveUntilTouchController's
/// thresholds.
/// When those thresholds are exceeded, the controller stops the movement.
class FTThresholdHelper {

public:
  /// Constructor.
  /// \param[in] useThresholdControl You can turn this whole object on and off.
  /// Useful if you don't use the MoveUntilTouchController and don't need to set
  /// these thresholds.
  /// \param[in] nodeHandle Handle of the ros node.
  /// \param[in] topicOverride manually specify FTThreshold Action Server
  FTThresholdHelper(bool useThresholdControl, ros::NodeHandle nodeHandle);

  /// Swaps the action client to a new server.
  /// Blocks until server is online.
  void swapTopic(const std::string &topic);

  /// Needs to be called before setting the first thresholds.
  /// Blocks until the threshold could be set successfully.
  /// Can be aborted with Ctrl-C.
  void init(bool retare = true, const std::string &topicOverride = "");

  /// Sets the MoveUntilTouchControllers Thresholds accordingly.
  /// Throws a runtime_error if we useThresholdControl and we are unable to set
  /// because of an error.
  /// \return True if the thresholds were set successfully or false if we
  /// experienced a timeout.
  bool setThresholds(std::string preset, bool retare = false);

  bool setThresholds(double forces, double torques, bool retare = false);

  // Collect until stop if numberOfDataPoints == 0
  bool startDataCollection(size_t numberOfDataPoints = 0);
  void stopDataCollection();

  bool isDataCollectionFinished();

  // Return false if !isDatacollectionFinished
  bool getDataAverage(Eigen::Vector3d &forceMean, Eigen::Vector3d &torqueMean);
  bool writeDataToFile(const std::string &fileName);

  // Return empty if !isDatacollectionFinished
  std::vector<double> getData();

private:
  bool mUseThresholdControl;
  ros::NodeHandle mNodeHandle;

  size_t mDataPointsToCollect = 0;
  std::mutex mDataCollectionMutex;
  std::vector<Eigen::Vector3d> mCollectedForces;
  std::vector<Eigen::Vector3d> mCollectedTorques;
  std::vector<ros::Time> mTimestamps;
  std::atomic<bool> mCollectingData;

  // \brief Gets data from the force/torque sensor
  ros::Subscriber mForceTorqueDataSub;

#ifdef REWD_CONTROLLERS_FOUND
  std::unique_ptr<rewd_controllers::FTThresholdClient> mFTThresholdClient;
#endif

  std::pair<double, double> getThresholdValues(std::string preset);

  /**
   * \brief Called whenever a new Force/Torque message arrives on the ros topic
   */
  void forceTorqueDataCallback(const geometry_msgs::WrenchStamped &msg);
};
} // namespace feeding

#endif
