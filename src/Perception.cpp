#include "feeding/Perception.hpp"

#include <aikido/perception/AssetDatabase.hpp>

#include "feeding/util.hpp"

namespace feeding {

//==============================================================================
Perception::Perception(
    aikido::planner::WorldPtr world,
    dart::dynamics::MetaSkeletonPtr adasMetaSkeleton,
    ros::NodeHandle nodeHandle)
  : mWorld(world), mNodeHandle(nodeHandle)
{
  std::string detectorDataURI
      = getRosParam<std::string>("/perception/detectorDataUri", mNodeHandle);
  std::string referenceFrameName
      = getRosParam<std::string>("/perception/referenceFrameName", mNodeHandle);
  std::string detectorTopicName
      = getRosParam<std::string>("/perception/detectorTopicName", mNodeHandle);

  const auto resourceRetriever
      = std::make_shared<aikido::io::CatkinResourceRetriever>();

  mObjDetector = std::unique_ptr<aikido::perception::PoseEstimatorModule>(
      new aikido::perception::PoseEstimatorModule(
          mNodeHandle,
          detectorTopicName,
          std::make_shared<aikido::perception::AssetDatabase>(
              resourceRetriever, detectorDataURI),
          resourceRetriever,
          referenceFrameName,
          aikido::robot::util::getBodyNodeOrThrow(
              *adasMetaSkeleton, referenceFrameName)));
}

//==============================================================================
bool Perception::perceiveFood(Eigen::Isometry3d& foodTransform)
{
  mObjDetector->detectObjects(
      mWorld,
      ros::Duration(
          getRosParam<double>("/perception/timeoutSeconds", mNodeHandle)));

  // just choose one for now
  std::string perceivedFoodName
      = getRosParam<std::string>("/perception/foodName", mNodeHandle);
  auto perceivedFood = mWorld->getSkeleton(perceivedFoodName);
  if (perceivedFood != nullptr)
  {
    foodTransform = Eigen::Isometry3d::Identity();
    foodTransform.translation()
        = perceivedFood->getBodyNode(0)->getWorldTransform().translation();
    return true;
  }
  else
  {
    return false;
  }
}
} // namespace feeding
