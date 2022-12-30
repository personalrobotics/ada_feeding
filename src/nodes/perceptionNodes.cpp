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

// Utility to Extract
class FrameFromObject : public BT::SyncActionNode {
public:
  FrameFromObject(const std::string &name, const BT::NodeConfig &config)
      : BT::SyncActionNode(name, config) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<DetectedObject>("object"),
            BT::InputPort<double>("fixed_x"),
            BT::InputPort<double>("fixed_y"),
            BT::InputPort<double>("fixed_z"),
            BT::InputPort<int>("align_idx"),
            BT::InputPort<double>("rotate_aligned"),
            BT::OutputPort<Eigen::Isometry3d>("frame")};
  }

  BT::NodeStatus tick() override {
    // Read Params
    auto objectInput = getInput<DetectedObject>("object");
    if (!objectInput) {
      return BT::NodeStatus::FAILURE;
    }
    DetectedObject obj = objectInput.value();
    Eigen::Isometry3d objTransform =
        obj.getMetaSkeleton()->getBodyNode(0)->getWorldTransform();
    Eigen::Isometry3d frame = Eigen::Isometry3d::Identity();
    frame.translation() = objTransform.translation();

    // Set Fixed X,Y,Z if provided
    if (getInput<double>("fixed_x"))
      frame.translation()[0] = getInput<double>("fixed_x").value();
    if (getInput<double>("fixed_y"))
      frame.translation()[1] = getInput<double>("fixed_y").value();
    if (getInput<double>("fixed_z"))
      frame.translation()[2] = getInput<double>("fixed_z").value();

    // "Flatten" orientation, so obj[idx] aligns ith world[idx]
    // We do this by taking the idx+1 axis of the object frame
    // and projecting it onto the idx+1/idx+2 plane of the world
    // frame. This is the (idx+1)-axis of the new frame
    auto alignInput = getInput<int>("align_idx");
    if (alignInput) {
      std::vector<Eigen::Vector3d> unitVecs{Eigen::Vector3d::UnitX(),
                                            Eigen::Vector3d::UnitY(),
                                            Eigen::Vector3d::UnitZ()};
      auto alignIdx = alignInput.value() % 3;

      Eigen::Vector3d flat =
          objTransform.rotation() * unitVecs[(alignIdx + 1) % 3];
      Eigen::AngleAxisd rotation = Eigen::AngleAxisd(
          atan2(flat[(alignIdx + 2) % 3], flat[(alignIdx + 1) % 3]),
          unitVecs[alignIdx % 3]);
      frame.linear() = frame.linear() * rotation;

      // Apply additional rotation about aligning idx if requested
      auto rot = getInput<double>("rotate_aligned");
      if (rot) {
        frame.linear() = frame.linear() *
                         Eigen::AngleAxisd(rot.value(), unitVecs[alignIdx % 3]);
      }
    }
    // Else just use object rotation
    else {
      frame.linear() = objTransform.linear();
    }

    setOutput<Eigen::Isometry3d>("frame", frame);
    return BT::NodeStatus::SUCCESS;
  }
};

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
  // Input Faces
  auto objectInput = self.getInput<DetectedObject>("face");
  if (!objectInput) {
    return BT::NodeStatus::FAILURE;
  }
  DetectedObject obj = objectInput.value();

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

  // Perception Utility
  factory.registerNodeType<FrameFromObject>("FrameFromObject");

  // Perception Functions
  factory.registerNodeType<PerceiveFn<kFOOD>>("PerceiveFood");
  factory.registerNodeType<PerceiveFn<kFACE>>("PerceiveFace");

  factory.registerSimpleAction("IsMouthOpen", IsMouthOpen,
                               {BT::InputPort<DetectedObject>("face")});

  factory.registerSimpleAction(
      "ClearPerceptionList", ClearList,
      {BT::OutputPort<std::vector<DetectedObject>>("target")});
}
static_block { feeding::registerNodeFn(&registerNodes); }

} // end namespace nodes
} // end namespace feeding
