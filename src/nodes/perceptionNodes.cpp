#include "feeding/nodes.hpp"
/**
 * Nodes for interfacing with aikido::perception
 **/

#include <Eigen/Core>
#include <behaviortree_cpp/behavior_tree.h>
#include <functional>
#include <iostream>
#include <vector>

#include <aikido/perception/AssetDatabase.hpp>
#include <aikido/perception/DetectedObject.hpp>
#include <aikido/perception/PoseEstimatorModule.hpp>
using aikido::perception::DetectedObject;

#include <libada/util.hpp>
using ada::util::getRosParam;

#include <yaml-cpp/exceptions.h>

namespace feeding {
namespace nodes {

// Perception Module
class AdaPerception {
public:
  std::vector<DetectedObject> perceiveFood() {
    std::lock_guard<std::mutex> lock(mFoodMutex);
    return currentFood;
  }

  std::vector<DetectedObject> perceiveFace() {
    std::lock_guard<std::mutex> lock(mFaceMutex);
    return currentFace;
  }

  void init(ros::NodeHandle *nh, ada::Ada *robot) {
    // Init basic vars
    mAda = robot;
    const auto resourceRetriever =
        std::make_shared<aikido::io::CatkinResourceRetriever>();

    // Read Private Parameters
    std::string detectorDataURI =
        getRosParam<std::string>("perception/detectorDataUri", *nh);
    std::string referenceFrameName =
        getRosParam<std::string>("perception/referenceFrameName", *nh);

    // Init Class Vars
    mAssetDatabase = std::make_shared<aikido::perception::AssetDatabase>(
        resourceRetriever, detectorDataURI);

    mFoodDetector = std::unique_ptr<aikido::perception::PoseEstimatorModule>(
        new aikido::perception::PoseEstimatorModule(
            *nh, "food_detector", mAssetDatabase, resourceRetriever,
            referenceFrameName,
            aikido::robot::util::getBodyNodeOrThrow(*(mAda->getMetaSkeleton()),
                                                    referenceFrameName)));

    mFaceDetector = std::unique_ptr<aikido::perception::PoseEstimatorModule>(
        new aikido::perception::PoseEstimatorModule(
            *nh, "face_detector", mAssetDatabase, resourceRetriever,
            referenceFrameName,
            aikido::robot::util::getBodyNodeOrThrow(*(mAda->getMetaSkeleton()),
                                                    referenceFrameName)));
    mPerceptionRate = getRosParam<double>("perception/rateSeconds", *nh);
    mPerceptionTimeout = getRosParam<double>("perception/timeoutSeconds", *nh);

    // Start perception loops
    mFoodTimer = nh->createTimer(
        ros::Duration(mPerceptionRate),
        std::bind(&AdaPerception::foodCallback, this, std::placeholders::_1));
    mFoodTimer.stop();
    mFaceTimer = nh->createTimer(
        ros::Duration(mPerceptionRate),
        std::bind(&AdaPerception::faceCallback, this, std::placeholders::_1));
    mFaceTimer.stop();
  }

  void enableTimers() {
    currentFood.clear();
    currentFace.clear();
    mFoodTimer.start();
    mFaceTimer.start();
  }

  void disableTimers() {
    mFoodTimer.stop();
    mFaceTimer.stop();
    currentFood.clear();
    currentFace.clear();
  }

private:
  // Timer Callbacks
  void foodCallback(const ros::TimerEvent &) {
    std::vector<DetectedObject> detectedObjects;
    mFoodDetector->detectObjects(mAda->getWorld(),
                                 ros::Duration(mPerceptionTimeout),
                                 ros::Time(0), &detectedObjects);
    std::lock_guard<std::mutex> lock(mFoodMutex);
    currentFood.clear();
    currentFood = detectedObjects;
  }

  void faceCallback(const ros::TimerEvent &) {
    std::vector<DetectedObject> detectedObjects;
    mFaceDetector->detectObjects(mAda->getWorld(),
                                 ros::Duration(mPerceptionTimeout),
                                 ros::Time(0), &detectedObjects);
    std::lock_guard<std::mutex> lock(mFaceMutex);
    currentFace.clear();
    currentFace = detectedObjects;
  }

  // Members
  ada::Ada *mAda;

  double mPerceptionTimeout, mPerceptionRate;
  ros::Timer mFoodTimer, mFaceTimer;

  std::vector<DetectedObject> currentFood;
  std::vector<DetectedObject> currentFace;

  // Aikido Perception Module
  std::unique_ptr<aikido::perception::PoseEstimatorModule> mFoodDetector;
  std::unique_ptr<aikido::perception::PoseEstimatorModule> mFaceDetector;
  std::shared_ptr<aikido::perception::AssetDatabase> mAssetDatabase;

  /// Manages access to currentFood
  mutable std::mutex mFoodMutex;
  /// Manages access to currentFace
  mutable std::mutex mFaceMutex;
};
static AdaPerception sPerception;

enum PerceiveType { kFOOD, kFACE };

template <PerceiveType T> class PerceiveFn : public BT::StatefulActionNode {

public:
  PerceiveFn(const std::string &name, const BT::NodeConfig &config)
      : BT::StatefulActionNode(name, config) {}

  static BT::PortsList providedPorts() {
    BT::PortsList ret = {
        BT::InputPort<double>("timeout"),
        BT::OutputPort<std::vector<DetectedObject>>("objects")};

    if (T == PerceiveType::kFOOD)
      ret.insert(BT::InputPort<std::string>("name_filter"));

    return ret;
  }

  BT::NodeStatus onStart() override {
    // Init Class Vars
    mStartTime = ros::Time::now();
    auto timeout = getInput<double>("timeout");
    mTimeout = timeout ? ros::Duration(timeout.value()) : ros::Duration(0);
    auto nameFilter = getInput<std::string>("name_filter");
    mNameFilter = nameFilter ? nameFilter.value() : "";

    sPerception.enableTimers();
    return onRunning();
  }

  BT::NodeStatus onRunning() override {
    // Detect Objects
    std::vector<DetectedObject> objects;
    switch (T) {
    case kFACE:
      objects = sPerception.perceiveFace();
      break;
    case kFOOD:
    default:
      objects = sPerception.perceiveFood();
    }

    if (objects.size() > 0) {
      sPerception.disableTimers();
      // Apply Name Filter
      if (mNameFilter != "") {
        objects.erase(std::remove_if(objects.begin(), objects.end(),
                                     [this](const DetectedObject &x) {
                                       return x.getName() != mNameFilter;
                                     }),
                      objects.end());
      }
      // Return success
      setOutput<std::vector<DetectedObject>>("objects", objects);
      return objects.size() > 0 ? BT::NodeStatus::SUCCESS
                                : BT::NodeStatus::FAILURE;
    } else if (ros::Time::now() - mStartTime >= mTimeout) {
      sPerception.disableTimers();
      // Return failure if beyond timeout
      return BT::NodeStatus::FAILURE;
    }

    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override { sPerception.disableTimers(); }

private:
  ros::Time mStartTime;
  ros::Duration mTimeout;
  std::string mNameFilter;
};

// Detect if mouth is open
BT::NodeStatus IsMouthOpen(BT::TreeNode &self) {
  // Input FAces
  auto objectInput = self.getInput<std::vector<DetectedObject>>("faces");
  if (!objectInput || objectInput.value().size() < 1) {
    return BT::NodeStatus::FAILURE;
  }
  // Just select the first object
  // TODO: more intelligent object selection
  DetectedObject obj = objectInput.value()[0];

  bool mouthOpen = false;
  try {
    auto yamlNode = obj.getYamlNode();
    if (yamlNode["mouth-status"].as<std::string>() == "open") {
      mouthOpen = true;
    }
  } catch (const YAML::Exception &e) { /* Do Nothing */
  }

  return mouthOpen ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

// Clear List of objects
BT::NodeStatus ClearList(BT::TreeNode &self) {
  self.setOutput("target", std::vector<DetectedObject>());
  return BT::NodeStatus::SUCCESS;
}

/// Node registration
static void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                          ada::Ada &robot) {
  sPerception.init(&nh, &robot);

  // Perception Functions
  factory.registerNodeType<PerceiveFn<kFOOD>>("PerceiveFood");
  factory.registerNodeType<PerceiveFn<kFACE>>("PerceiveFace");

  factory.registerSimpleAction(
      "IsMouthOpen", IsMouthOpen,
      {BT::InputPort<std::vector<DetectedObject>>("faces")});

  factory.registerSimpleAction(
      "ClearPerceptionList", ClearList,
      {BT::OutputPort<std::vector<DetectedObject>>("target")});
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
