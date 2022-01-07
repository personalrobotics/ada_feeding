#include "feeding/FoodItem.hpp"

#include <yaml-cpp/exceptions.h>

namespace feeding {

//==============================================================================
FoodItem::FoodItem(std::string name, std::string uid,
                   dart::dynamics::MetaSkeletonPtr metaSkeleton,
                   AcquisitionAction action, double score)
    : mName(name), mUid(uid), mMetaSkeleton(metaSkeleton), mAction(action),
      mScore(score) {
  if (!mMetaSkeleton)
    throw std::invalid_argument("MetaSkeleton is nullptr.");
}

FoodItem::FoodItem(std::string name, std::string uid,
                   dart::dynamics::MetaSkeletonPtr metaSkeleton,
                   AcquisitionAction action, double score,
                   const YAML::Node info)
    : mName(name), mUid(uid), mMetaSkeleton(metaSkeleton), mAction(action),
      mScore(score), mExtraInfo(info) {
  if (!mMetaSkeleton)
    throw std::invalid_argument("MetaSkeleton is nullptr.");
}

//==============================================================================
Eigen::Isometry3d FoodItem::getPose() const {
  return mMetaSkeleton->getBodyNode(0)->getWorldTransform();
}

//==============================================================================
std::string FoodItem::getName() const { return mName; }

//==============================================================================
std::string FoodItem::getUid() const { return mUid; }

//==============================================================================
dart::dynamics::MetaSkeletonPtr FoodItem::getMetaSkeleton() const {
  return mMetaSkeleton;
}

//==============================================================================
AcquisitionAction const *FoodItem::getAction() const { return &mAction; }

//==============================================================================
void FoodItem::setAction(int actionNum) {
  TiltStyle tiltStyle(TiltStyle::NONE);
  // Create New Acquisition Action
  switch (actionNum / 2) {
  case 1:
    tiltStyle = TiltStyle::VERTICAL;
    break;
  case 2:
    tiltStyle = TiltStyle::ANGLED;
    break;
  default:
    tiltStyle = TiltStyle::NONE;
  }
  auto rotation = (actionNum % 2 == 0) ? 0.0 : M_PI / 2.0;

  // TODO: check if rotation and tilt angle should change
  AcquisitionAction action(tiltStyle, rotation, 0.0, Eigen::Vector3d(0, 0, -1));
  mAction = action;
}

//==============================================================================
double FoodItem::getScore() const { return mScore; }

//==============================================================================
YAML::Node FoodItem::getExtraInfo() const { return mExtraInfo; }

} // namespace feeding