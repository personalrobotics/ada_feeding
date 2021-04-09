#include "feeding/ranker/SuccessRateRanker.hpp"

#include <dart/common/StlHelpers.hpp>

#include "feeding/AcquisitionAction.hpp"

namespace feeding {

//==============================================================================
void SuccessRateRanker::sort(
    std::vector<std::unique_ptr<FoodItem>>& items) const
{
  // Descending since score is the succes rate.
  TargetFoodRanker::sort(items, SORT_ORDER::DESCENDING);
}

//==============================================================================
std::unique_ptr<FoodItem> SuccessRateRanker::createFoodItem(
    const aikido::perception::DetectedObject& item,
    const Eigen::Isometry3d& forqueTransform) const
{
  double successRate = item.getInfoByKey<double>("score");
  std::string itemAction = item.getInfoByKey<std::string>("action");
  if (StringToTiltStyle.find(itemAction) == StringToTiltStyle.end())
  {
    std::stringstream ss;
    ss << "Action [" << itemAction << "] not recognized." << std::endl;
    throw std::invalid_argument(ss.str());
  }
  TiltStyle tiltStyle = StringToTiltStyle.at(itemAction);
  double rotation = item.getInfoByKey<double>("rotation");

  // TODO: Make AcquisitionAction deterministic on tiltStyle?
  AcquisitionAction action(tiltStyle, rotation, 0.0, Eigen::Vector3d(0, 0, -1));

  auto itemPose = item.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
  return std::make_unique<FoodItem>(
      item.getName(),
      item.getUid(),
      item.getMetaSkeleton(),
      action,
      successRate);
}

} // namespace feeding
