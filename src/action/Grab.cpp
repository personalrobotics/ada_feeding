#include "feeding/action/Grab.hpp"

namespace feeding {
namespace action {

//==============================================================================
void grabFood(
    const std::shared_ptr<ada::Ada>& ada,
    const std::shared_ptr<Workspace>& workspace)
{
  if (!workspace->getDefaultFoodItem())
  {
    workspace->addDefaultFoodItemAtPose(
        ada->getHand()->getEndEffectorBodyNode()->getTransform());
  }
  ada->getHand()->grab(workspace->getDefaultFoodItem());
}

//==============================================================================
void ungrabAndDeleteFood(
    const std::shared_ptr<ada::Ada>& ada,
    const std::shared_ptr<Workspace>& workspace)
{
  ada->getHand()->ungrab();
  workspace->deleteFood();
}

} // namespace action
} // namespace feeding
