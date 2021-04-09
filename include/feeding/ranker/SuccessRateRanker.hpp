#ifndef FEEDING_SUCCESSRATERANKER_HPP_
#define FEEDING_SUCCESSRATERANKER_HPP_

#include "feeding/ranker/TargetFoodRanker.hpp"

namespace feeding {

/// Ranks items based on their predicted success rate
/// This ranker expects that DetectedObject contains the following
/// keys: {"score": double, "action": string, "rotation": double}
/// The string values for "action" is those in \c AcquisitionAction.
class SuccessRateRanker : public TargetFoodRanker
{
public:
  /// Returns a sorted list of items.
  /// \param[in] items List of food items.
  /// \param[out] items List of food items.
  void sort(std::vector<std::unique_ptr<FoodItem>>& items) const override;

  // Documentation inherited.
  std::unique_ptr<FoodItem> createFoodItem(
      const aikido::perception::DetectedObject& item,
      const Eigen::Isometry3d& forqueTransform) const override;
};

} // namespace feeding

#endif
