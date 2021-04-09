#include "feeding/ranker/ShortestDistanceRanker.hpp"

#include <iterator>

#include <dart/common/StlHelpers.hpp>

#include "feeding/AcquisitionAction.hpp"
#include "feeding/util.hpp"

namespace feeding {

//==============================================================================
void ShortestDistanceRanker::sort(
    std::vector<std::unique_ptr<FoodItem>>& items) const
{
  // Ascending since score is the distance.
  TargetFoodRanker::sort(items, SORT_ORDER::ASCENDING);
}

//==============================================================================
std::unique_ptr<FoodItem> ShortestDistanceRanker::createFoodItem(
    const aikido::perception::DetectedObject& item,
    const Eigen::Isometry3d& forqueTransform) const
{
  TiltStyle tiltStyle(TiltStyle::NONE);
  double rotation = 0.0;

  // Get Ideal Action Per Food Item
  auto it = std::find(FOOD_NAMES.begin(), FOOD_NAMES.end(), item.getName());
  if (it != FOOD_NAMES.end())
  {
    int actionNum = BEST_ACTIONS[std::distance(FOOD_NAMES.begin(), it)];
    switch (actionNum / 2)
    {
      case 1:
        tiltStyle = TiltStyle::VERTICAL;
        break;
      case 2:
        tiltStyle = TiltStyle::ANGLED;
        break;
      default:
        tiltStyle = TiltStyle::NONE;
    }
    rotation = (actionNum % 2 == 0) ? 0.0 : M_PI / 2.0;
  }

  // TODO: check if rotation and tilt angle should change
  AcquisitionAction action(tiltStyle, rotation, 0.0, Eigen::Vector3d(0, 0, -1));

  auto itemPose = item.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
  double distance = getDistance(itemPose, forqueTransform);

  return std::make_unique<FoodItem>(
      item.getName(),
      item.getUid(),
      item.getMetaSkeleton(),
      action,
      distance,
      item.getYamlNode());
}

} // namespace feeding
