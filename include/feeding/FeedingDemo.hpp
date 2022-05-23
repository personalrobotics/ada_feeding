#ifndef FEEDING_FEEDINGDEMO_HPP_
#define FEEDING_FEEDINGDEMO_HPP_

#include <aikido/distance/ConfigurationRanker.hpp>
#include <aikido/planner/World.hpp>
#include <aikido/rviz/InteractiveMarkerViewer.hpp>
#include <aikido/rviz/TSRMarker.hpp>
#include <ros/ros.h>

#include <libada/Ada.hpp>

#include "feeding/AcquisitionAction.hpp"
#include "feeding/FTThresholdHelper.hpp"
#include "feeding/TargetItem.hpp"
#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"
#include "feeding/perception/PerceptionServoClient.hpp"
#include "feeding/ranker/TargetFoodRanker.hpp"

namespace feeding {

/// The FeedingDemo class is responsible for
/// - The robot (loading + control)
/// - The workspace
///
/// It contains functions that are very specialized for the feeding demo,
/// like moveInFrontOfPerson(). It uses the robot, workspace info and ros
/// parameters
/// to accomplish these tasks.
class FeedingDemo {

public:
  /// Constructor for the Feeding Demo.
  /// Takes care of setting up the robot and the workspace
  /// \param[in] adaReal True if the real robot is used, false it's running in
  /// simulation.
  /// \param[in] useFTSensing turns the FTSensor and the
  /// MoveUntilTouchController on and off
  /// \param[in] useVisualServo If true, perception servo is used.
  /// \param[in] allowFreeRotation, If true, items specified as rotationFree
  /// get rotational freedom.
  /// \param[in] nodeHandle Handle of the ros node.
  FeedingDemo(bool adaReal, std::shared_ptr<ros::NodeHandle> nodeHandle,
              bool useFTSensingToStopTrajectories, bool useVisualServo,
              bool allowFreeRotation,
              std::shared_ptr<FTThresholdHelper> ftThresholdHelper = nullptr,
              bool autoContinueDemo = false);

  /// Destructor for the Feeding Demo.
  /// Also shuts down the trajectory controllers.
  ~FeedingDemo();

  void setPerception(std::shared_ptr<Perception> perception);

  /// Gets the Node Handle
  std::shared_ptr<ros::NodeHandle> getNodeHandle();

  /// Gets the aikido world
  aikido::planner::WorldPtr getWorld();

  /// Gets the workspace
  std::shared_ptr<Workspace> getWorkspace();

  /// Gets Ada
  std::shared_ptr<ada::Ada> getAda();

  /// Gets Ada replication in Simulation
  std::shared_ptr<ada::Ada> getAdaSimulation();

  /// Determines if Demo is Real or Sim
  bool isAdaReal();

  /// Gets the transform of the default food object (defined in Workspace)
  /// Valid only for simulation mode
  Eigen::Isometry3d getDefaultFoodTransform();

  aikido::rviz::InteractiveMarkerViewerPtr getViewer();

  void waitForUser(const std::string &prompt);

  aikido::constraint::dart::CollisionFreePtr getCollisionConstraint();

  aikido::constraint::dart::CollisionFreePtr
  getCollisionConstraintWithWallFurtherBack();

  /// Resets the environmnet.
  void reset();

  std::shared_ptr<FTThresholdHelper> getFTThresholdHelper();

  Eigen::Isometry3d getPlateEndEffectorTransform() const;

  // bool moveWithEndEffectorTwist(
  //   const Eigen::Vector6d& twists,
  //   double durations = 1.0,
  //   bool respectCollision = true);

  std::vector<std::string> mFoodNames;
  std::vector<std::string> mRotationFreeFoodNames;
  std::vector<std::string> mTiltFoodNames;
  std::vector<double> mSkeweringForces;
  std::unordered_map<std::string, double> mFoodSkeweringForces;
  std::unordered_map<std::string, int> mPickUpAngleModes;

  std::unordered_map<std::string, double> mPlateTSRParameters;
  std::unordered_map<std::string, double> mFoodTSRParameters;
  std::unordered_map<std::string, double> mPersonTSRParameters;
  double mMoveOufOfFoodLength;

  double mPlanningTimeout;
  int mMaxNumTrials;
  int mBatchSize;
  int mMaxNumBatches;
  int mNumMaxIterations;
  double mEndEffectorOffsetPositionTolerance;
  double mEndEffectorOffsetAngularTolerance;
  std::chrono::milliseconds mWaitTimeForFood;
  std::chrono::milliseconds mWaitTimeForPerson;
  Eigen::Vector6d mVelocityLimits;

  double mForkHolderAngle;
  std::vector<double> mForkHolderTranslation;
  Eigen::Vector3d mTiltOffset;

  double mTableHeight;

private:
  /// Attach food to forque
  void grabFoodWithForque();

  /// Detach food from forque and remove it from the aikido world.
  void ungrabAndDeleteFood();

  bool mIsFTSensingEnabled;
  bool mAdaReal;
  bool mAutoContinueDemo;
  bool mVisualServo;
  bool mAllowRotationFree;
  std::shared_ptr<ros::NodeHandle> mNodeHandle;
  std::shared_ptr<Perception> mPerception;
  std::shared_ptr<FTThresholdHelper> mFTThresholdHelper;

  aikido::planner::WorldPtr mWorld;

  aikido::planner::WorldPtr mWorldSimulation;

  std::shared_ptr<ada::Ada> mAda;

  std::shared_ptr<ada::Ada> mAdaSimulation;

  aikido::statespace::dart::MetaSkeletonStateSpacePtr mArmSpace;
  std::shared_ptr<Workspace> mWorkspace;
  aikido::constraint::dart::CollisionFreePtr mCollisionFreeConstraint;
  aikido::constraint::dart::CollisionFreePtr
      mCollisionFreeConstraintWithWallFurtherBack;

  std::unique_ptr<PerceptionServoClient> mServoClient;

  std::vector<aikido::rviz::TSRMarkerPtr> tsrMarkers;

  aikido::rviz::InteractiveMarkerViewerPtr mViewer;
  aikido::rviz::InteractiveMarkerViewerPtr mViewerSim;
  
  aikido::rviz::FrameMarkerPtr frameMarker;
  aikido::rviz::TrajectoryMarkerPtr trajectoryMarkerPtr;
};
} // namespace feeding

#endif
