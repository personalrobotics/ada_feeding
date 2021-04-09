#ifndef FEEDING_TARGETFOODRANKER_HPP_
#define FEEDING_TARGETFOODRANKER_HPP_

#include <vector>

#include <Eigen/Core>

#include "feeding/FoodItem.hpp"

namespace feeding {

enum SORT_ORDER
{
  ASCENDING = 0,
  DESCENDING = 1
};

// Base class for ranking target food items.
class TargetFoodRanker
{
public:
  /// Returns a sorted list of items.
  /// \param[in] items List of food items.
  /// \param[out] items List of food items.
  virtual void sort(std::vector<std::unique_ptr<FoodItem>>& items) const = 0;

  virtual std::unique_ptr<FoodItem> createFoodItem(
      const aikido::perception::DetectedObject& item,
      const Eigen::Isometry3d& forqueTransform) const = 0;

protected:
  /// Returns a sorted list of items.
  /// \param[in] items List of food items.
  /// \param[out] items List of food items.
  void sort(
      std::vector<std::unique_ptr<FoodItem>>& items, SORT_ORDER order) const;
};

} // namespace feeding

#endif
