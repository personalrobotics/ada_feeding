#include "feeding/perception/PerceptionServoClient.hpp"

#include <chrono>

#include <aikido/constraint/Satisfied.hpp>
#include <aikido/planner/ConfigurationToConfiguration.hpp>
#include <aikido/planner/SnapConfigurationToConfigurationPlanner.hpp>
#include <aikido/planner/kunzretimer/KunzRetimer.hpp>
#include <aikido/planner/parabolic/ParabolicTimer.hpp>
#include <aikido/statespace/dart/MetaSkeletonStateSaver.hpp>
#include <aikido/trajectory/util.hpp>

#include <libada/util.hpp>

#include "feeding/util.hpp"

#define THRESHOLD 10.0 // s to wait for good frame

using ada::util::getRosParam;
using aikido::constraint::Satisfied;
using aikido::planner::ConfigurationToConfiguration;
using aikido::planner::SnapConfigurationToConfigurationPlanner;
using aikido::planner::kunzretimer::computeKunzTiming;
using aikido::statespace::dart::MetaSkeletonStateSaver;
using aikido::trajectory::concatenate;
using aikido::trajectory::createPartialTrajectory;
using aikido::trajectory::findTimeOfClosestStateOnTrajectory;
using aikido::trajectory::Interpolated;
using aikido::trajectory::Spline;
using aikido::trajectory::SplinePtr;
using aikido::trajectory::TrajectoryPtr;
using aikido::trajectory::UniqueInterpolatedPtr;
using aikido::trajectory::UniqueSplinePtr;

namespace feeding {

namespace {

Eigen::VectorXd getSymmetricLimits(
    const Eigen::VectorXd& lowerLimits, const Eigen::VectorXd& upperLimits)
{
  assert(
      static_cast<std::size_t>(lowerLimits.size())
      == static_cast<std::size_t>(upperLimits.size()));

  std::size_t limitSize = static_cast<std::size_t>(lowerLimits.size());
  Eigen::VectorXd symmetricLimits(limitSize);
  for (std::size_t i = 0; i < limitSize; ++i)
  {
    symmetricLimits[i] = std::min(-lowerLimits[i], upperLimits[i]);
  }
  return symmetricLimits;
}

} // namespace

//==============================================================================
PerceptionServoClient::PerceptionServoClient(
    const ::ros::NodeHandle* node,
    boost::function<Eigen::Isometry3d(void)> getTransform,
    aikido::statespace::dart::ConstMetaSkeletonStateSpacePtr
        metaSkeletonStateSpace,
    std::shared_ptr<ada::Ada> ada,
    ::dart::dynamics::MetaSkeletonPtr metaSkeleton,
    ::dart::dynamics::BodyNodePtr bodyNode,
    std::shared_ptr<aikido::control::TrajectoryExecutor> trajectoryExecutor,
    aikido::constraint::dart::CollisionFreePtr collisionFreeConstraint,
    double perceptionUpdateTime,
    float goalPrecision,
    double planningTimeout,
    double endEffectorOffsetPositionTolerance,
    double endEffectorOffsetAngularTolerance,
    bool servoFood,
    const Eigen::Vector6d& velocityLimits)
  : mNodeHandle(*node, "perceptionServo")
  , mGetTransform(getTransform)
  , mMetaSkeletonStateSpace(std::move(metaSkeletonStateSpace))
  , mAda(std::move(ada))
  , mMetaSkeleton(std::move(metaSkeleton))
  , mBodyNode(bodyNode)
  , mTrajectoryExecutor(trajectoryExecutor)
  , mCollisionFreeConstraint(collisionFreeConstraint)
  , mPerceptionUpdateTime(perceptionUpdateTime)
  , mCurrentTrajectory(nullptr)
  , mGoalPrecision(goalPrecision)
  , mPlanningTimeout(planningTimeout)
  , mEndEffectorOffsetPositionTolerance(endEffectorOffsetPositionTolerance)
  , mEndEffectorOffsetAngularTolerance(endEffectorOffsetAngularTolerance)
  , mServoFood(servoFood)
  , mIsRunning(false)
{
  mNonRealtimeTimer = mNodeHandle.createTimer(
      ros::Duration(mPerceptionUpdateTime),
      &PerceptionServoClient::nonRealtimeCallback,
      this,
      false,
      false);

  // update Max velocity and acceleration
  mMaxAcceleration = getSymmetricLimits(
      mMetaSkeleton->getAccelerationLowerLimits(),
      mMetaSkeleton->getAccelerationUpperLimits());

  mOriginalPose = mBodyNode->getTransform();
  mOriginalConfig = mMetaSkeleton->getPositions();

  mVelocityLimits = Eigen::VectorXd(velocityLimits.size());
  for (std::size_t i = 0; i < velocityLimits.size(); ++i)
    mVelocityLimits[i] = 0.8 * velocityLimits[i];
}

//==============================================================================
PerceptionServoClient::~PerceptionServoClient()
{
  if (mTrajectoryExecutor)
  {
    mTrajectoryExecutor->cancel();
  }

  if (mExec.valid())
  {
    mExec.wait_for(std::chrono::duration<int, std::milli>(1000));
  }

  mNonRealtimeTimer.stop();
  mSub.shutdown();
  ROS_WARN("shutting down perception servo client");
}

//==============================================================================
void PerceptionServoClient::start()
{
  ROS_INFO("Servoclient started");
  mExecutionDone = false;
  mNonRealtimeTimer.start();
  mNotFailed = false;
  mStartTime = std::chrono::system_clock::now();
  mLastSuccess = mStartTime;
}

//==============================================================================
void PerceptionServoClient::stop()
{
  mTimerMutex.lock();
  // Always cancel the executing trajectory when quitting
  mTrajectoryExecutor->cancel();
  mNonRealtimeTimer.stop();
  mIsRunning = false;
  mTimerMutex.unlock();
}

//==============================================================================
bool PerceptionServoClient::wait(double timelimit)
{
  double elapsedTime = 0.0;
  std::chrono::time_point<std::chrono::system_clock> startTime
      = std::chrono::system_clock::now();
  while (elapsedTime < timelimit && !mExecutionDone)
  {

    // sleep a while
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(
                      std::chrono::system_clock::now() - startTime)
                      .count();
  }
  stop();
  if (elapsedTime >= 20)
    ROS_INFO_STREAM(
        "Timeout " << timelimit << " reached for PerceptionServoClient");

  return mNotFailed;
}

//==============================================================================
bool PerceptionServoClient::isRunning()
{
  return mIsRunning;
}

//==============================================================================
void PerceptionServoClient::nonRealtimeCallback(const ros::TimerEvent& event)
{
  std::cout << event.last_expected << " " << event.last_real << " "
            << event.current_expected << " " << event.current_real << " "
            << event.profile.last_duration << std::endl;
  std::cout << "Entering " << __LINE__ << std::endl;
  std::cout << mPerceptionUpdateTime << std::endl;
  if (mExecutionDone || !mTimerMutex.try_lock())
  {
    std::cout << "Done? " << mExecutionDone << std::endl;
    return;
  }

  Eigen::Isometry3d goalPose;
  if (updatePerception(goalPose))
  {
    std::cout << "Entering " << __LINE__ << std::endl;
    mLastSuccess = std::chrono::system_clock::now();
    // Generate a new reference trajectory to the goal pose
    if (mExecutionDone)
    {
      std::cout << "Entering " << __LINE__ << std::endl;
      ROS_WARN_STREAM("Completed");
      mTimerMutex.unlock();
      return;
    }
    mCurrentTrajectory = planToGoalPose(goalPose);

    if (!mCurrentTrajectory)
    {
      std::cout << "Entering " << __LINE__ << std::endl;
      ROS_WARN_STREAM("Failed to get trajectory");
      mTimerMutex.unlock();
      return;
    }
    // Save current pose
    mOriginalPose = mBodyNode->getTransform();
    mOriginalConfig = mMetaSkeleton->getPositions();

    if (mIsRunning && mExec.valid()
        && (mExec.wait_for(std::chrono::duration<int, std::milli>(0))
            != std::future_status::ready))
    {
      std::cout << "Entering " << __LINE__ << std::endl;
      ROS_INFO_STREAM("Cancel the current trajectory");
      mTrajectoryExecutor->cancel();
      mExec.wait();
    }

    // Execute the new reference trajectory
    ROS_INFO_STREAM("Sending a new trajectory");
    // std::cout << "Running: " << mExec.wait_for(std::chrono::duration<int,
    // std::milli>(0))
    //         != std::future_status::ready << std::endl;
    mExec = mTrajectoryExecutor->execute(mCurrentTrajectory);
    mIsRunning = true;
  }
  else
  {
    double sinceLast
        = std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::system_clock::now() - mLastSuccess)
              .count();
    if (sinceLast > THRESHOLD)
    {
      std::cout << "Entering " << __LINE__ << std::endl;
      ROS_WARN("Lost perception for too long. Reporting failure...");
      mExecutionDone = true;
      mNotFailed = false;
    }
    else
    {
      std::cout << "Entering " << __LINE__ << std::endl;
      ROS_WARN_STREAM("Perception Failed. Since Last: " << sinceLast);
    }
  }
  mTimerMutex.unlock();
}

//==============================================================================
bool PerceptionServoClient::updatePerception(Eigen::Isometry3d& goalPose)
{
  // update new goal Pose
  ROS_INFO_STREAM("entering update");
  Eigen::Isometry3d endEffectorTransform(Eigen::Isometry3d::Identity());
  Eigen::Matrix3d rotation(
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())
      * Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  endEffectorTransform.linear() = rotation;

  Eigen::Isometry3d pose;
  try
  {
    pose = mGetTransform();
    if (mRemoveRotation)
      removeRotation(pose);
  }
  catch (std::runtime_error& e)
  {
    ROS_WARN_STREAM(e.what());
    return false;
  }

  goalPose = pose * endEffectorTransform;
  Eigen::Isometry3d currentPose = mBodyNode->getTransform();

  // Step 1: Plan from current pose to goal pose.
  Eigen::Vector3d vectorToGoalPose
      = goalPose.translation() - currentPose.translation();

  // if (vectorToGoalPose.norm() < 0.15)
  // {
  // goalPose = mPreviousGoalPose;
  // return true;
  // }

  // mPreviousGoalPose = goalPose;

  std::cout << "Goal Pose " << goalPose.translation() << std::endl;
  if (goalPose.translation().z() < -0.1)
  {
    ROS_WARN_STREAM("Food is way too low, z " << goalPose.translation()[2]);
    return false;
  }
  ROS_INFO_STREAM("leaving update");
  return true;
}

//==============================================================================
SplinePtr PerceptionServoClient::planToGoalPose(
    const Eigen::Isometry3d& goalPose)
{

  // for (std::size_t i = 0; i < mVelocityLimits.size(); ++i)
  // mVelocityLimits[i] = 0.9*mVelocityLimits[i];

  // using dart::dynamics::InverseKinematics;
  // using aikido::statespace::dart::MetaSkeletonStateSaver;
  // using aikido::planner::ConfigurationToConfiguration;
  // using aikido::planner::SnapConfigurationToConfigurationPlanner;
  // using aikido::planner::kunzretimer::computeKunzTiming;
  // using aikido::trajectory::Interpolated;
  // using aikido::trajectory::Spline;

  // Eigen::Isometry3d currentPose = mBodyNode->getTransform();

  // aikido::trajectory::Spline* spline1 = nullptr;
  // aikido::trajectory::Spline* spline2 = nullptr;

  // Eigen::VectorXd currentConfig = mMetaSkeleton->getPositions();
  // Eigen::Vector3d direction1
  //     = currentPose.translation() - mOriginalPose.translation();

  // aikido::trajectory::TrajectoryPtr trajectory1 = nullptr;
  // if (direction1.norm() > 1e-2)
  // {
  //   ROS_WARN_STREAM("CAME TO FIRST IF");
  //   MetaSkeletonStateSaver saver1(mMetaSkeleton);
  //   mMetaSkeleton->setPositions(mOriginalConfig);

  //   // using vectorfield planner directly because Ada seems to update the
  //   state
  //   // otherwise
  //   // trajectory1 =
  //   mAdaMover->planToEndEffectorOffset(direction1.normalized(),
  //   // direction1.norm());
  //   auto originalState = mMetaSkeletonStateSpace->createState();
  //   mMetaSkeletonStateSpace->convertPositionsToState(
  //       mOriginalConfig, originalState);

  //   ROS_INFO_STREAM("Servoing plan to end effector offset 1 state: " <<
  //   mMetaSkeleton->getPositions().matrix().transpose());
  //   ROS_INFO_STREAM("Servoing plan to end effector offset 1 direction: " <<
  //   direction1.normalized().matrix().transpose() << ",  length: " <<
  //   direction1.norm());

  //   auto satisfiedConstraint =
  //   std::make_shared<Satisfied>(mMetaSkeletonStateSpace);

  //   std::chrono::time_point<std::chrono::system_clock> startTime =
  //   std::chrono::system_clock::now(); trajectory1 = planToEndEffectorOffset(
  //       mMetaSkeletonStateSpace,
  //       *originalState,
  //       mMetaSkeleton,
  //       mBodyNode,
  //       satisfiedConstraint,
  //       direction1.normalized(),
  //       0.0, // direction1.norm() - 0.001,
  //       direction1.norm() + 0.004,
  //       0.08,
  //       0.32,
  //       0.001,
  //       1e-3,
  //       1e-2,
  //       std::chrono::duration<double>(5));

  //   ROS_INFO_STREAM("Planning 1 took " <<
  //   std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now()
  //   - startTime).count());

  //   if (trajectory1 == nullptr)
  //     throw std::runtime_error("Failed in finding the first half");
  //   spline1 = dynamic_cast<aikido::trajectory::Spline*>(trajectory1.get());
  // }

  // Eigen::Vector3d direction2 = goalPose.translation() -
  // currentPose.translation(); if (direction2.norm() < 0.002)
  // {
  //   ROS_WARN_STREAM("CAME TO SECOND IF");
  //   ROS_INFO("Visual servoing is finished because goal was position
  //   reached."); mExecutionDone = true; return nullptr;
  // }

  // aikido::trajectory::TrajectoryPtr trajectory2 = nullptr;
  // if (spline1)
  // {
  //   ROS_WARN_STREAM("CAME TO THIRD IF");
  //   MetaSkeletonStateSaver saver2(mMetaSkeleton);
  //   auto endState = mMetaSkeletonStateSpace->createState();
  //   auto endTime = spline1->getEndTime();
  //   spline1->evaluate(spline1->getEndTime(), endState);
  //   mMetaSkeletonStateSpace->setState(mMetaSkeleton.get(), endState);

  //   auto satisfiedConstraint =
  //   std::make_shared<aikido::constraint::Satisfied>(mMetaSkeletonStateSpace);

  //   aikido::planner::Planner::Result result;
  //   std::chrono::time_point<std::chrono::system_clock> startTime =
  //   std::chrono::system_clock::now(); trajectory2 = planToEndEffectorOffset(
  //       mMetaSkeletonStateSpace,
  //       *endState,
  //       mMetaSkeleton,
  //       mBodyNode,
  //       satisfiedConstraint,
  //       direction2.normalized(),
  //       0.0, // std::min(direction2.norm(), 0.2) - 0.001,
  //       std::min(direction2.norm(), 0.0) + 0.1,
  //       0.01,
  //       0.04,
  //       0.001,
  //       1e-3,
  //       1e-3,
  //       std::chrono::duration<double>(5),
  //       &result);
  //   // ROS_INFO_STREAM("Planning 2 took " <<
  //   std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now()
  //   - startTime).count());

  //   if (trajectory2 == nullptr)
  //   {
  //     std::cout << "RESULT : " << result.getMessage() << std::endl;
  //     throw std::runtime_error("Failed in finding the second half");
  //   }
  //   spline2 = dynamic_cast<aikido::trajectory::Spline*>(trajectory2.get());
  //   if (spline2 == nullptr)
  //     return nullptr;

  //   // AVK modification
  //   std::chrono::time_point<std::chrono::system_clock> timingStartTime =
  //   std::chrono::system_clock::now(); auto concatenatedTraj = concatenate(
  //     // *spline1,
  //     // *spline2
  //     *dynamic_cast<Interpolated*>(spline1),
  //     *dynamic_cast<Interpolated*>(spline2)
  //     );
  //   if (concatenatedTraj == nullptr)
  //     return nullptr;

  //   auto timedTraj = computeKunzTiming(
  //       *concatenatedTraj, mMaxVelocity, mMaxAcceleration, 1e-2, 9e-3);

  //   if (timedTraj == nullptr)
  //     return nullptr;

  //   // AVK Addition.
  //   double distance;
  //   auto state = mMetaSkeletonStateSpace->createState();
  //   mMetaSkeletonStateSpace->convertPositionsToState(
  //     mMetaSkeleton->getPositions(), state);
  //   double refTime
  //       = findTimeOfClosestStateOnTrajectory(*timedTraj, state, distance,
  //       0.01);

  //   auto partialTimedTraj = createPartialTrajectory(*timedTraj, refTime);

  //   return std::move(partialTimedTraj);
  // }
  // else
  // {
  //   ROS_WARN_STREAM("CAME TO ELSE");
  //   aikido::trajectory::TrajectoryPtr trajectory2 =
  //   planEndEffectorOffset(direction2.normalized(),
  //       0.03);

  //   if (trajectory2 == nullptr)
  //     throw std::runtime_error("Failed in finding the traj");

  //   spline2 = dynamic_cast<aikido::trajectory::Spline*>(trajectory2.get());
  //   if (spline2 == nullptr)
  //     return nullptr;

  //   auto timedTraj = computeKunzTiming(
  //       *dynamic_cast<Interpolated*>(spline2), mMaxVelocity,
  //       mMaxAcceleration, 1e-2, 3e-3);

  //   if (timedTraj == nullptr)
  //     return nullptr;

  //   return std::move(timedTraj);
  // }

  ROS_INFO_STREAM("1");

  Eigen::Isometry3d currentPose = mBodyNode->getTransform();

  // Step 1: Plan from current pose to goal pose.
  Eigen::Vector3d vectorToGoalPose
      = goalPose.translation() - currentPose.translation();

  std::cout << "GOAL POSE DISTANCE " << vectorToGoalPose.norm() << std::endl;

  if (vectorToGoalPose.norm() < mGoalPrecision)
  {
    ROS_WARN("Visual servoing is finished because goal was position reached.");
    mExecutionDone = true;
    mNotFailed = true;
    return nullptr;
  }

  if (vectorToGoalPose[2] > 0 && mServoFood)
  {
    ROS_WARN(
        "Visual servoing is finished because goal is above the current pose");
    mExecutionDone = true;
    mNotFailed = true;
    return nullptr;
  }

  ROS_INFO_STREAM("2");

  auto trajToGoal = planEndEffectorOffset(vectorToGoalPose);
  if (!trajToGoal)
  {
    ROS_WARN_STREAM("Plan failed");
    return nullptr;
  }

  ROS_INFO_STREAM("3");

  // Step 2: Plan from original pose to current pose.
  Eigen::Vector3d vectorFromOriginalToCurrent(
      currentPose.translation() - mOriginalPose.translation());

  // || (goalPose.translation() - mPreviousGoalPose.translation()).norm() <
  // 0.01)
  UniqueSplinePtr timedTraj;
  if (vectorFromOriginalToCurrent.norm() < 0.001)
  {
    ROS_WARN_STREAM("Returning trajectory from current to goal");
    timedTraj = computeKunzTiming(
        *dynamic_cast<Interpolated*>(trajToGoal.get()),
        mVelocityLimits,
        mMaxAcceleration,
        1e-2,
        3e-3);

    if (!timedTraj)
      ROS_WARN_STREAM("Concatenation &/ timing failed");

    return timedTraj;
  }

  else
  {
    ROS_WARN_STREAM("Computing the second part of the trrajectory");
    auto originalState = mMetaSkeletonStateSpace->createState();
    mMetaSkeletonStateSpace->convertPositionsToState(
        mOriginalConfig, originalState);

    ROS_INFO_STREAM("5");

    // auto trajOriginalToCurrent = ada->planToOffset(
    //     mMetaSkeletonStateSpace,
    //     *originalState,
    //     mMetaSkeleton,
    //     mBodyNode,
    //     std::make_shared<Satisfied>(mMetaSkeletonStateSpace),
    //     vectorFromOriginalToCurrent.normalized(),
    //     0.0,
    //     vectorFromOriginalToCurrent.norm(),
    //     0.08,
    //     0.32,
    //     0.001,
    //     1e-3,
    //     1e-2,
    //     std::chrono::duration<double>(5));
    auto trajOriginalToCurrent = mAda->planToOffset(
      mAda->getEndEffectorBodyNode()->getName(),
      vectorFromOriginalToCurrent,
      mAda->getArm()->getWorldCollisionConstraint());

    ROS_INFO_STREAM("6");

    if (!trajOriginalToCurrent)
      throw std::runtime_error("Failed to generate first half of trajectory");

    ROS_WARN_STREAM(
        "Interpolator 1: " << (dynamic_cast<Interpolated*>(
                                   trajOriginalToCurrent.get()))
                                  ->getInterpolator());
    ROS_WARN_STREAM(
        "Interpolator 2: "
        << (dynamic_cast<Interpolated*>(trajToGoal.get()))->getInterpolator());
    ROS_WARN_STREAM(
        "Statespace 1: " << (dynamic_cast<Interpolated*>(
                                 trajOriginalToCurrent.get()))
                                ->getStateSpace());
    ROS_WARN_STREAM(
        "Statespace 2: "
        << (dynamic_cast<Interpolated*>(trajToGoal.get()))->getStateSpace());

    // Step 3: Concatenate the two trajectories.
    auto concatenatedTraj = concatenate(
        *dynamic_cast<Interpolated*>(trajOriginalToCurrent.get()),
        *dynamic_cast<Interpolated*>(trajToGoal.get()));

    ROS_INFO_STREAM("7");

    timedTraj = computeKunzTiming(
        *dynamic_cast<Interpolated*>(concatenatedTraj.get()),
        mVelocityLimits,
        mMaxAcceleration,
        1e-2,
        9e-3);

    ROS_INFO_STREAM("8");
  }

  if (!timedTraj)
  {
    ROS_WARN_STREAM("Concatenation &/ timing failed");
    return nullptr;
  }

  ROS_INFO_STREAM("9");
  // Start from the closest point on the trajectory.
  timedTraj = createPartialTimedTrajectoryFromCurrentConfig(timedTraj.get());
  // Eigen::VectorXd segStartVel(mVelocityLimits.size());
  // timedTraj->evaluateDerivative(timedTraj->getStartTime(), 1, segStartVel);
  // std::cout << "The start velocity of the next trajectory " <<
  // segStartVel.transpose() << std::endl;

  ROS_INFO_STREAM("10");
  return timedTraj;
}

//==============================================================================
TrajectoryPtr PerceptionServoClient::planEndEffectorOffset(
    const Eigen::Vector3d& goalDirection, double threshold)
{
  if (goalDirection.norm() < 1e-3)
    return nullptr;

  // return mAda->planArmToEndEffectorOffset(
  //     goalDirection.normalized(),
  //     // 0.15,
  //     std::min(goalDirection.norm(), threshold),
  //     nullptr,
  //     mPlanningTimeout,
  //     mEndEffectorOffsetPositionTolerance,
  //     mEndEffectorOffsetAngularTolerance);
  return mAda->planToOffset(
      mAda->getEndEffectorBodyNode()->getName(),
      goalDirection);
}

//==============================================================================
UniqueSplinePtr
PerceptionServoClient::createPartialTimedTrajectoryFromCurrentConfig(
    const Spline* trajectory)
{
  double distance;
  auto state = mMetaSkeletonStateSpace->createState();
  mMetaSkeletonStateSpace->convertPositionsToState(
      mMetaSkeleton->getPositions(), state);

  double refTime
      = findTimeOfClosestStateOnTrajectory(*trajectory, state, distance, 0.01);

  if (distance > 1.0)
  {
    ROS_WARN_STREAM("Distance too far " << distance);
    return nullptr;
  }

  std::cout << "Shorted distance " << distance << " at " << refTime
            << std::endl;

  // Start 0.3 sec forward since the robot has been moving.
  refTime += 0.8;
  if (refTime > trajectory->getEndTime())
  {
    ROS_WARN_STREAM("Robot already reached end of trajectory.");
    return nullptr;
  }
  auto traj = createPartialTrajectory(*trajectory, refTime);
  if (!traj || traj->getDuration() < 1e-5)
  {
    ROS_WARN_STREAM("Trajectory duration too short.");
    return nullptr;
  }
  return traj;
}

} // namespace feeding
