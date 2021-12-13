#ifndef FEEDING_PERCEPTIONSERVOCLIENT_HPP_
#define FEEDING_PERCEPTIONSERVOCLIENT_HPP_

#include <mutex>

#include <aikido/control/ros/RosTrajectoryExecutor.hpp>
#include <aikido/rviz/InteractiveMarkerViewer.hpp>
#include <aikido/statespace/dart/MetaSkeletonStateSpace.hpp>
#include <aikido/trajectory/Spline.hpp>
#include <boost/optional.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

#include <libada/Ada.hpp>

#include "feeding/perception/Perception.hpp"

namespace feeding {

class PerceptionServoClient {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PerceptionServoClient(
      const ::ros::NodeHandle *node,
      boost::function<Eigen::Isometry3d(void)> getTransform,
      aikido::statespace::dart::ConstMetaSkeletonStateSpacePtr
          metaSkeletonStateSpace,
      std::shared_ptr<ada::Ada> ada,
      ::dart::dynamics::MetaSkeletonPtr metaSkeleton,
      ::dart::dynamics::BodyNodePtr bodyNode,
      std::shared_ptr<aikido::control::TrajectoryExecutor> trajectoryExecutor,
      aikido::constraint::dart::CollisionFreePtr collisionFreeConstraint,
      double perceptionUpdateTime, float goalPrecision, double planningTimeout,
      double endEffectorOffsetPositionTolerance,
      double endEffectorOffsetAngularTolerance, bool servoFood,
      const Eigen::Vector6d &velocityLimits);

  virtual ~PerceptionServoClient();

  void start();

  void stop();

  bool isRunning();

  bool wait(double timelimit);

protected:
  void nonRealtimeCallback(const ros::TimerEvent &event);

  bool updatePerception(Eigen::Isometry3d &goalPose);

  aikido::trajectory::SplinePtr
  planEndEffectorOffset(const Eigen::Isometry3d &goalPose);

  aikido::trajectory::SplinePtr
  planToGoalPose(const Eigen::Isometry3d &goalPose);

  aikido::trajectory::TrajectoryPtr
  planEndEffectorOffset(const Eigen::Vector3d &goalDirection,
                        double threshold = 0.1);

  aikido::trajectory::UniqueSplinePtr
  createPartialTimedTrajectoryFromCurrentConfig(
      const aikido::trajectory::Spline *trajectory);

  ::ros::NodeHandle mNodeHandle;
  boost::function<Eigen::Isometry3d(void)> mGetTransform;
  /// Meta skeleton state space.
  aikido::statespace::dart::ConstMetaSkeletonStateSpacePtr
      mMetaSkeletonStateSpace;

  /// Meta Skeleton
  ::dart::dynamics::MetaSkeletonPtr mMetaSkeleton;

  /// BodyNode
  ::dart::dynamics::BodyNodePtr mBodyNode;

  std::shared_ptr<aikido::control::TrajectoryExecutor> mTrajectoryExecutor;

  double mPerceptionUpdateTime;

  aikido::trajectory::SplinePtr mCurrentTrajectory;

  std::future<void> mExec;

  ros::Timer mNonRealtimeTimer;

  Eigen::VectorXd mMaxVelocity;
  Eigen::VectorXd mMaxAcceleration;

  Eigen::VectorXd mCurrentPosition;

  Eigen::Isometry3d mOriginalPose;
  Eigen::Isometry3d mPreviousGoalPose;
  Eigen::VectorXd mOriginalConfig;

  aikido::constraint::dart::CollisionFreePtr mCollisionFreeConstraint;

  std::vector<dart::dynamics::SimpleFramePtr> mFrames;
  std::vector<aikido::rviz::FrameMarkerPtr> mFrameMarkers;
  bool mExecutionDone;
  bool mIsRunning;
  bool mNotFailed;
  bool mServoFood;

  ros::Subscriber mSub;
  std::mutex mJointStateUpdateMutex;
  std::mutex mTimerMutex;

  std::shared_ptr<ada::Ada> mAda;

  std::chrono::time_point<std::chrono::system_clock> mStartTime;
  std::chrono::time_point<std::chrono::system_clock> mLastSuccess;

  float mGoalPrecision = 0.01;

  double mPlanningTimeout;
  double mEndEffectorOffsetPositionTolerance;
  double mEndEffectorOffsetAngularTolerance;

  bool mRemoveRotation;
  Eigen::VectorXd mVelocityLimits;
};
} // namespace feeding

#endif // FEEDING_PERCEPTIONSERVOCLIENT_HPP_
