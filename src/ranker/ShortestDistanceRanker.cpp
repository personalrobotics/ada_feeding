#include "feeding/ranker/ShortestDistanceRanker.hpp"

#include <iterator>

#include <dart/common/StlHelpers.hpp>

#include "feeding/AcquisitionAction.hpp"
#include "feeding/util.hpp"

namespace feeding {

//==============================================================================
void ShortestDistanceRanker::sort(
    std::vector<std::unique_ptr<FoodItem>> &items) const {
  // Ascending since score is the distance.
  TargetFoodRanker::sort(items, SORT_ORDER::ASCENDING);
}

//==============================================================================
std::unique_ptr<FoodItem> ShortestDistanceRanker::createFoodItem(
    const aikido::perception::DetectedObject &item,
    const Eigen::Isometry3d &forqueTransform) const {
  TiltStyle tiltStyle(TiltStyle::NONE);
  double rotation = 0.0;

  // Get Action From DetectedObject
  std::string itemAction = item.getInfoByKey<std::string>("action");
  if (StringToTiltStyle.find(itemAction) != StringToTiltStyle.end()) {
    TiltStyle tiltStyle = StringToTiltStyle.at(itemAction);
    double rotation = item.getInfoByKey<double>("rotation");
  }

  // TODO: Make AcquisitionAction deterministic on tiltStyle?
  AcquisitionAction action(tiltStyle, rotation, 0.0, Eigen::Vector3d(0, 0, -1));

  auto itemPose = item.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
  double distance = getDistance(itemPose, forqueTransform);

  return std::make_unique<FoodItem>(item.getName(), item.getUid(),
                                    item.getMetaSkeleton(), action, distance,
                                    item.getYamlNode());
}

} // namespace feeding
