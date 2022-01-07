#ifndef FEEDING_FOODITEM_HPP_
#define FEEDING_FOODITEM_HPP_

#include <aikido/common/pointers.hpp>
#include <aikido/perception/DetectedObject.hpp>
#include <dart/dart.hpp>
#include <yaml-cpp/exceptions.h>

#include "feeding/AcquisitionAction.hpp"

namespace feeding {

AIKIDO_DECLARE_POINTERS(FoodItem)

class FoodItem {
public:
  FoodItem(std::string name, std::string uid,
           dart::dynamics::MetaSkeletonPtr metaSkeleton,
           AcquisitionAction action, double score);

  FoodItem(std::string name, std::string uid,
           dart::dynamics::MetaSkeletonPtr metaSkeleton,
           AcquisitionAction action, double score, const YAML::Node info);

  Eigen::Isometry3d getPose() const;

  std::string getName() const;

  std::string getUid() const;

  dart::dynamics::MetaSkeletonPtr getMetaSkeleton() const;

  AcquisitionAction const *getAction() const;
  void setAction(int actionNum);

  double getScore() const;

  YAML::Node getExtraInfo() const;

private:
  const std::string mName;

  const std::string mUid; // unique id necessary for tracking

  const dart::dynamics::MetaSkeletonPtr mMetaSkeleton;

  AcquisitionAction mAction;

  double mScore;

  const YAML::Node mExtraInfo;
};

} // namespace feeding

#endif
