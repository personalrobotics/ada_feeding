#include "feeding/ranker/TargetFoodRanker.hpp"

#include <algorithm>

#include "feeding/util.hpp"

namespace feeding {

//==============================================================================
void TargetFoodRanker::sort(
    std::vector<std::unique_ptr<FoodItem>>& items, SORT_ORDER order) const
{
  if (order == SORT_ORDER::ASCENDING)
  {
    std::sort(
        items.begin(),
        items.end(),
        [&](std::unique_ptr<FoodItem>& v1, std::unique_ptr<FoodItem>& v2) {
          return v1->getScore() <= v2->getScore();
        });
  }
  else
  {
    std::sort(
        items.begin(),
        items.end(),
        [&](std::unique_ptr<FoodItem>& v1, std::unique_ptr<FoodItem>& v2) {
          return v1->getScore() >= v2->getScore();
        });
  }

  std::cout << "Sort result: " << order << std::endl;

  for (auto& item : items)
    std::cout << item->getName() << " " << item->getScore() << std::endl;
}
} // namespace feeding
