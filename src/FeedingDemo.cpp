#include "feeding/FeedingDemo.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <aikido/robot/util.hpp>
#include <aikido/rviz/TrajectoryMarker.hpp>
#include <boost/optional.hpp>

#include <libada/util.hpp>

#include "feeding/FoodItem.hpp"
#include "feeding/util.hpp"

using ada::util::createIsometry;
using ada::util::getRosParam;
using aikido::constraint::dart::CollisionFreePtr;

const bool TERMINATE_AT_USER_PROMPT = true;

static const std::size_t MAX_NUM_TRIALS = 3;
static const double DEFAULT_VELOCITY_LIM = 0.2;
static const double inf = std::numeric_limits<double>::infinity();

namespace feeding {

//==============================================================================
FeedingDemo::FeedingDemo(bool adaReal,
                         std::shared_ptr<ros::NodeHandle> nodeHandle,
                         bool useFTSensingToStopTrajectories,
                         bool useVisualServo, bool allowFreeRotation,
                         std::shared_ptr<FTThresholdHelper> ftThresholdHelper,
                         bool autoContinueDemo)
    : mAdaReal(adaReal), mNodeHandle(nodeHandle),
      mFTThresholdHelper(ftThresholdHelper), mVisualServo(useVisualServo),
      mAllowRotationFree(allowFreeRotation),
      mAutoContinueDemo(autoContinueDemo),
      mIsFTSensingEnabled(useFTSensingToStopTrajectories) {
  mWorld = std::make_shared<aikido::planner::World>("feeding");

  std::string armTrajectoryExecutor = mIsFTSensingEnabled
                                          ? "move_until_touch_topic_controller"
                                          : "rewd_trajectory_controller";

  mAda = std::make_shared<ada::Ada>(!mAdaReal, mWorld);
  // mArmSpace = mAda->getArm()->getStateSpace();

  Eigen::Isometry3d robotPose = createIsometry(
      getRosParam<std::vector<double>>("/ada/baseFramePose", *mNodeHandle));

  mWorkspace =
      std::make_shared<Workspace>(mWorld, robotPose, mAdaReal, *mNodeHandle);

  // visualization
  mViewer = std::make_shared<aikido::rviz::InteractiveMarkerViewer>(
      getRosParam<std::string>("/visualization/topicName", *mNodeHandle),
      getRosParam<std::string>("/visualization/baseFrameName", *mNodeHandle),
      mWorld);
  mViewer->setAutoUpdate(true);

  if (mAdaReal) {
    mAda->startTrajectoryControllers();
  }

  // Load the named configurations if available
  auto retriever = std::make_shared<aikido::io::CatkinResourceRetriever>();
  std::string nameConfigs =
      getRosParam<std::string>("/af_named_configs", *mNodeHandle);
  if (nameConfigs != "") // correctly write this
  {
    auto rootNode = aikido::io::loadYAML(nameConfigs, retriever);
    if (rootNode["hand"]) {
      mAda->getHandRobot()->setNamedConfigurations(
          aikido::robot::util::parseYAMLToNamedConfigurations(
              rootNode["hand"]));
    }
    if (rootNode["arm"]) {
      mAda->getArm()->setNamedConfigurations(
          aikido::robot::util::parseYAMLToNamedConfigurations(rootNode["arm"]));
    }
  }

  mFoodNames =
      getRosParam<std::vector<std::string>>("/foodItems/names", *mNodeHandle);
  mSkeweringForces =
      getRosParam<std::vector<double>>("/foodItems/forces", *mNodeHandle);
  mRotationFreeFoodNames = getRosParam<std::vector<std::string>>(
      "/rotationFree/names", *mNodeHandle);
  mTiltFoodNames =
      getRosParam<std::vector<std::string>>("/tiltFood/names", *mNodeHandle);
  auto pickUpAngleModes = getRosParam<std::vector<int>>(
      "/foodItems/pickUpAngleModes", *mNodeHandle);

  for (int i = 0; i < mFoodNames.size(); i++) {
    mFoodSkeweringForces[mFoodNames[i]] = mSkeweringForces[i];
    mPickUpAngleModes[mFoodNames[i]] = pickUpAngleModes[i];
  }

  mPlateTSRParameters["height"] =
      getRosParam<double>("/feedingDemo/heightAbovePlate", *mNodeHandle);
  mPlateTSRParameters["horizontalTolerance"] = getRosParam<double>(
      "/planning/tsr/horizontalToleranceAbovePlate", *mNodeHandle);
  mPlateTSRParameters["verticalTolerance"] = getRosParam<double>(
      "/planning/tsr/verticalToleranceAbovePlate", *mNodeHandle);
  mPlateTSRParameters["rotationTolerance"] = getRosParam<double>(
      "/planning/tsr/rotationToleranceAbovePlate", *mNodeHandle);

  mFoodTSRParameters["height"] =
      getRosParam<double>("/feedingDemo/heightAboveFood", *mNodeHandle);
  mFoodTSRParameters["heightInto"] =
      getRosParam<double>("/feedingDemo/heightIntoFood", *mNodeHandle);
  mFoodTSRParameters["horizontalTolerance"] = getRosParam<double>(
      "/planning/tsr/horizontalToleranceNearFood", *mNodeHandle);
  mFoodTSRParameters["verticalTolerance"] = getRosParam<double>(
      "/planning/tsr/verticalToleranceNearFood", *mNodeHandle);
  mFoodTSRParameters["rotationTolerance"] = getRosParam<double>(
      "/planning/tsr/rotationToleranceNearFood", *mNodeHandle);
  mFoodTSRParameters["tiltTolerance"] =
      getRosParam<double>("/planning/tsr/tiltToleranceNearFood", *mNodeHandle);
  mMoveOufOfFoodLength =
      getRosParam<double>("/feedingDemo/moveOutofFood", *mNodeHandle);

  mPlanningTimeout =
      getRosParam<double>("/planning/timeoutSeconds", *mNodeHandle);
  mMaxNumTrials = getRosParam<int>("/planning/maxNumberOfTrials", *mNodeHandle);
  mBatchSize = getRosParam<int>("/planning/batchSize", *mNodeHandle);
  mMaxNumBatches = getRosParam<int>("/planning/maxNumberOfBatches", *mNodeHandle);
  mNumMaxIterations = getRosParam<int>("/planning/numMaxIterations", *mNodeHandle);

  mEndEffectorOffsetPositionTolerance = getRosParam<double>(
      "/planning/endEffectorOffset/positionTolerance", *mNodeHandle),
  mEndEffectorOffsetAngularTolerance = getRosParam<double>(
      "/planning/endEffectorOffset/angularTolerance", *mNodeHandle);

  mWaitTimeForFood = std::chrono::milliseconds(
      getRosParam<int>("/feedingDemo/waitMillisecsAtFood", *mNodeHandle));
  mWaitTimeForPerson = std::chrono::milliseconds(
      getRosParam<int>("/feedingDemo/waitMillisecsAtPerson", *mNodeHandle));

  mPersonTSRParameters["distance"] =
      getRosParam<double>("/feedingDemo/distanceToPerson", *mNodeHandle);
  mPersonTSRParameters["horizontalTolerance"] = getRosParam<double>(
      "/planning/tsr/horizontalToleranceNearPerson", *mNodeHandle);
  mPersonTSRParameters["verticalTolerance"] = getRosParam<double>(
      "/planning/tsr/verticalToleranceNearPerson", *mNodeHandle);

  mForkHolderAngle =
      getRosParam<double>("/study/forkHolderAngle", *mNodeHandle);
  mForkHolderTranslation = getRosParam<std::vector<double>>(
      "/study/forkHolderTranslation", *mNodeHandle);

  std::vector<double> tiltOffsetVector =
      getRosParam<std::vector<double>>("/study/tiltOffset", *mNodeHandle);
  mTiltOffset = Eigen::Vector3d(tiltOffsetVector[0], tiltOffsetVector[1],
                                tiltOffsetVector[2]);

  std::vector<double> velocityLimits =
      getRosParam<std::vector<double>>("/study/velocityLimits", *mNodeHandle);
  mVelocityLimits = DEFAULT_VELOCITY_LIM * Eigen::Vector6d::Ones();

  // If /study/velocityLimits size < 6
  // assume it is only setting the limits
  // for the first few joints.
  for (std::size_t i = 0; i < velocityLimits.size(); ++i) {
    mVelocityLimits[i] = velocityLimits[i];
  }

  mTableHeight = getRosParam<double>("/study/tableHeight", *mNodeHandle);
}

//==============================================================================
FeedingDemo::~FeedingDemo() {
  if (mAdaReal) {
    // wait for a bit so controller actually stops moving
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    mAda->stopTrajectoryControllers();
  }
}

//==============================================================================
void FeedingDemo::setPerception(std::shared_ptr<Perception> perception) {
  mPerception = perception;
}

//==============================================================================
std::shared_ptr<ros::NodeHandle> FeedingDemo::getNodeHandle() {
  return mNodeHandle;
}

//==============================================================================
aikido::planner::WorldPtr FeedingDemo::getWorld() { return mWorld; }

//==============================================================================
std::shared_ptr<Workspace> FeedingDemo::getWorkspace() { return mWorkspace; }

//==============================================================================
std::shared_ptr<ada::Ada> FeedingDemo::getAda() { return mAda; }

//==============================================================================
bool FeedingDemo::isAdaReal() { return mAdaReal; }

//==============================================================================
CollisionFreePtr FeedingDemo::getCollisionConstraint() {
  return mCollisionFreeConstraint;
}

//==============================================================================
CollisionFreePtr FeedingDemo::getCollisionConstraintWithWallFurtherBack() {
  return mCollisionFreeConstraintWithWallFurtherBack;
}

//==============================================================================
Eigen::Isometry3d FeedingDemo::getDefaultFoodTransform() {
  return mWorkspace->getDefaultFoodItem()
      ->getRootBodyNode()
      ->getWorldTransform();
}

//==============================================================================
aikido::rviz::InteractiveMarkerViewerPtr FeedingDemo::getViewer() {
  return mViewer;
}

//==============================================================================
void FeedingDemo::reset() { mWorkspace->reset(); }

//==============================================================================
std::shared_ptr<FTThresholdHelper> FeedingDemo::getFTThresholdHelper() {
  return mFTThresholdHelper;
}

//==============================================================================
Eigen::Isometry3d FeedingDemo::getPlateEndEffectorTransform() const {
  // TODO: remove hardcoded transform for plate
  Eigen::Isometry3d eeTransform;
  Eigen::Matrix3d rot;
  rot << 1., 0., 0., 0., -1, 0., 0., 0., -1;
  eeTransform.linear() =
      rot *
      Eigen::Matrix3d(Eigen::AngleAxisd(M_PI * 0.5, Eigen::Vector3d::UnitZ()));
  eeTransform.translation() =
      Eigen::Vector3d(0, 0, mPlateTSRParameters.at("height"));

  return eeTransform;
}
} // namespace feeding
