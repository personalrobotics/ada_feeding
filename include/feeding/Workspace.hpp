#ifndef FEEDING_WORKSPACE_HPP_
#define FEEDING_WORKSPACE_HPP_

#include <aikido/planner/World.hpp>
#include <feeding/util.hpp>
#include <ros/ros.h>

namespace feeding {

/// The Workspace deals with everything in the aikido world
/// that is not a robot or something perceived by the camera.
class Workspace
{

public:
  /// Constructor of the Workspace.
  /// Fills the aikido world with stuff.
  /// Only loads the defaultFoodItem if the demo is run in simulation (because
  /// otherwise we will perceive the food).
  /// Since the robotPose needs to be in the origin of the aikido world,
  /// the placement of all objects depends on the robotPose on the table.
  /// \param[in] world The aikido world.
  /// \param[in] robotPose The pose of the robot relative to the workspace.
  /// \param[in] adaReal True if the real robot is used.
  /// \param[in] nodeHandle Handle of the ros node.
  Workspace(
      aikido::planner::WorldPtr world,
      const Eigen::Isometry3d& robotPose,
      bool adaReal,
      ros::NodeHandle nodeHandle);

  /// Gets the plate
  dart::dynamics::ConstSkeletonPtr getPlate() const;

  /// Gets the table
  dart::dynamics::ConstSkeletonPtr getTable() const;

  /// Gets the workspace environment
  dart::dynamics::ConstSkeletonPtr getWorkspaceEnvironment() const;

  dart::dynamics::ConstSkeletonPtr getWorkspaceEnvironmentWithWallFurtherBack()
      const;

  /// Gets the default food item
  dart::dynamics::SkeletonPtr getDefaultFoodItem() const;

  /// Gets the mannequin
  dart::dynamics::ConstSkeletonPtr getPerson() const;

  /// Gets the wheelchair
  dart::dynamics::ConstSkeletonPtr getWheelchair() const;

  Eigen::Isometry3d getPersonPose() const;

  /// Removes the default food item from the world.
  void deleteFood();

  void addDefaultFoodItemAtPose(const Eigen::Isometry3d& pose);

  /// Resets the environmnet.
  void reset();

private:
  ros::NodeHandle mNodeHandle;

  aikido::planner::WorldPtr mWorld;

  dart::dynamics::SkeletonPtr mPlate;
  dart::dynamics::SkeletonPtr mTable;
  dart::dynamics::SkeletonPtr mWorkspaceEnvironment;
  dart::dynamics::SkeletonPtr mWorkspaceEnvironmentWithWallFurtherBack;
  dart::dynamics::SkeletonPtr mDefaultFoodItem;
  dart::dynamics::SkeletonPtr mPerson;
  dart::dynamics::SkeletonPtr mWheelchair;

  Eigen::Isometry3d mRobotPose;
  Eigen::Isometry3d mPersonPose;

  /// Takes a skeleton pointer, fills it with a new skeleton and adds that to
  /// the world.
  /// \param[out] skeleton The skeleton pointer where we want to store the
  /// loaded skeleton.
  /// \param[in] name The name of the object that should be loaded.
  /// \param[in] robotPose The pose of the robot relative to the workspace.
  void addToWorld(
      dart::dynamics::SkeletonPtr& skeleton,
      const std::string& name,
      const Eigen::Isometry3d& robotPose);

  /// TODO: docstring
  void addToWorldAtPose(
      dart::dynamics::SkeletonPtr& skeleton,
      const std::string& name,
      const Eigen::Isometry3d& pose);
};
} // namespace feeding

#endif
