#include "feeding/action/DetectAndMoveAboveFood.hpp"

#include <chrono>
#include <thread>

#include <yaml-cpp/exceptions.h>

#include <libada/util.hpp>

#include "feeding/action/MoveAboveFood.hpp"
#include "feeding/util.hpp"

using ada::util::getRosParam;

namespace feeding {
namespace action {

std::unique_ptr<FoodItem>
detectAndMoveAboveFood(const std::shared_ptr<Perception> &perception,
                       const std::string &foodName, double rotationTolerance,
                       FeedingDemo *feedingDemo, double *angleGuess,
                       int actionOverride) {
  bool useSound = feedingDemo->mUseSound;

  std::vector<std::unique_ptr<FoodItem>> candidateItems;
  while (true) {
    // Perception returns the list of good candidates, any one of them is good.
    // Multiple candidates are preferrable since planning may fail.
    candidateItems = perception->perceiveFood(foodName);

    if (candidateItems.size() == 0) {
      // if (useSound) 
      //   talk("I can't find that food. Try putting it on the plate.");
      ROS_WARN_STREAM(
          "Failed to detect any food. Please place food on the plate.");
    } else
      break;

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  ROS_INFO_STREAM("Detected " << candidateItems.size() << " " << foodName);

  bool moveAboveSuccessful = false;

  std::unique_ptr<FoodItem> ret;

  for (auto &item : candidateItems) {
    // actionOverride = 5;

    if (actionOverride >= 0) {
      // Overwrite action in item
      item->setAction(actionOverride);
    }

    auto action = item->getAction();

    std::cout << "Tilt style " << action->getTiltStyle() << std::endl;
    moveAboveSuccessful = moveAboveFood(
        item->getName(), item->getPose(), action->getRotationAngle(),
        action->getTiltStyle(), rotationTolerance, feedingDemo, angleGuess);
    if (!moveAboveSuccessful) {
      ROS_INFO_STREAM("Failed to move above " << item->getName());
      if (useSound) 
        talk("Sorry, I'm having a little trouble moving. Let's try again.");
      return nullptr;
    }

    perception->setFoodItemToTrack(item.get());
    ret = std::move(item);
    break;
  }

  if (!moveAboveSuccessful) {
    ROS_ERROR("Failed to move above any food.");
    if (useSound) 
      talk(
          "Sorry, I'm having a little trouble moving. Mind if I get a little "
          "help?");
    return nullptr;
  }

  return ret;
}
} // namespace action
} // namespace feeding
