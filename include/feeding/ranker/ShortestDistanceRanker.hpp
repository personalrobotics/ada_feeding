#ifndef FEEDING_SHORTESTDISTANCERANKER_HPP_
#define FEEDING_SHORTESTDISTANCERANKER_HPP_

#include "feeding/FoodItem.hpp"
#include "feeding/ranker/TargetFoodRanker.hpp"

namespace feeding {

/// Ranks items based on their distance to the endeffector
class ShortestDistanceRanker : public TargetFoodRanker {
public:
  /// Returns a sorted list of items.
  /// \param[in] items List of food items.
  /// \param[out] items List of food items.
  void sort(std::vector<std::unique_ptr<FoodItem>> &items) const override;

  std::unique_ptr<FoodItem>
  createFoodItem(const aikido::perception::DetectedObject &item,
                 const Eigen::Isometry3d &forqueTransform) const override;
};

} // namespace feeding

#endif
