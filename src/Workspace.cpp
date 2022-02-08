#include "feeding/Workspace.hpp"

#include <aikido/io/CatkinResourceRetriever.hpp>
#include <aikido/io/util.hpp>

#include <libada/util.hpp>

using ada::util::createIsometry;
using ada::util::getRosParam;

namespace feeding {

//==============================================================================
Workspace::Workspace(aikido::planner::WorldPtr world,
                     const Eigen::Isometry3d &robotPose, bool adaReal,
                     ros::NodeHandle nodeHandle)
    : mNodeHandle(nodeHandle), mWorld(world), mRobotPose(robotPose) {
  addToWorld(mPlate, "plate", robotPose);
  addToWorld(mTable, "table", robotPose);
  addToWorld(mWheelchair, "wheelchair", Eigen::Isometry3d::Identity());
  addToWorld(mPerson, "person", Eigen::Isometry3d::Identity());
  mPersonPose = mPerson->getRootBodyNode()->getWorldTransform();
  reset();
}

//==============================================================================
void Workspace::reset() {
  if (mDefaultFoodItem)
    deleteFood();
}
//==============================================================================
void Workspace::addDefaultFoodItemAtPose(const Eigen::Isometry3d &pose) {
  addToWorldAtPose(mDefaultFoodItem, "defaultFoodItem", pose);
  mDefaultFoodItem->getRootBodyNode()->setCollidable(false);
}

//==============================================================================
void Workspace::addToWorld(dart::dynamics::SkeletonPtr &skeleton,
                           const std::string &name,
                           const Eigen::Isometry3d &robotPose) {
  Eigen::Isometry3d pose =
      robotPose.inverse() * createIsometry(getRosParam<std::vector<double>>(
                                "/" + name + "/pose", mNodeHandle));
  addToWorldAtPose(skeleton, name, pose);
}

//==============================================================================
void Workspace::addToWorldAtPose(dart::dynamics::SkeletonPtr &skeleton,
                                 const std::string &name,
                                 const Eigen::Isometry3d &pose) {
  const auto resourceRetriever =
      std::make_shared<aikido::io::CatkinResourceRetriever>();
  std::string urdfUri =
      getRosParam<std::string>("/" + name + "/urdfUri", mNodeHandle);
  skeleton = loadSkeletonFromURDF(resourceRetriever, urdfUri, pose);
  mWorld->addSkeleton(skeleton);
}

//==============================================================================
void Workspace::deleteFood() {
  auto freeJoint = dynamic_cast<dart::dynamics::FreeJoint *>(
      mDefaultFoodItem->getRootJoint());

  if (!freeJoint)
    throw std::runtime_error(
        "Unable to cast Skeleton's root joint to FreeJoint.");

  // for some reason the visualisation doesn't disappear properly
  Eigen::Isometry3d nowhere = Eigen::Isometry3d::Identity();
  nowhere.translation() = Eigen::Vector3d(0, 0, -10000);
  freeJoint->setTransform(nowhere);
  // mWorld->removeSkeleton(mDefaultFoodItem);
  mDefaultFoodItem = nullptr;
}

//==============================================================================
dart::dynamics::ConstSkeletonPtr Workspace::getPlate() const { return mPlate; }

//==============================================================================
dart::dynamics::ConstSkeletonPtr Workspace::getTable() const { return mTable; }

//==============================================================================
dart::dynamics::ConstSkeletonPtr Workspace::getWorkspaceEnvironment() const {
  return mWorkspaceEnvironment;
}

//==============================================================================
dart::dynamics::ConstSkeletonPtr
Workspace::getWorkspaceEnvironmentWithWallFurtherBack() const {
  return mWorkspaceEnvironmentWithWallFurtherBack;
}

//==============================================================================
dart::dynamics::SkeletonPtr Workspace::getDefaultFoodItem() const {
  return mDefaultFoodItem;
}

//==============================================================================
dart::dynamics::ConstSkeletonPtr Workspace::getPerson() const {
  return mPerson;
}

//==============================================================================
Eigen::Isometry3d Workspace::getPersonPose() const { return mPersonPose; }

//==============================================================================
dart::dynamics::ConstSkeletonPtr Workspace::getWheelchair() const {
  return mWheelchair;
}
} // namespace feeding
