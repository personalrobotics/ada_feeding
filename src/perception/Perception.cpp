#include "feeding/perception/Perception.hpp"

#include <algorithm>

#include <Eigen/Eigenvalues>
#include <aikido/perception/AssetDatabase.hpp>
#include <aikido/perception/DetectedObject.hpp>
#include <aikido/robot/util.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/topic.h>
#include <tf_conversions/tf_eigen.h>
#include <yaml-cpp/exceptions.h>

#include <libada/util.hpp>

#include "feeding/FoodItem.hpp"
#include "feeding/util.hpp"

using ada::util::getRosParam;
using aikido::perception::DetectedObject;

namespace feeding {

//==============================================================================
Perception::Perception(
    aikido::planner::WorldPtr world,
    std::shared_ptr<ada::Ada> ada,
    dart::dynamics::MetaSkeletonPtr adaMetaSkeleton,
    std::shared_ptr<ros::NodeHandle> nodeHandle,
    std::shared_ptr<TargetFoodRanker> ranker,
    float faceZOffset,
    bool removeRotationForFood)
  : mWorld(world)
  , mAda(ada)
  , mAdaMetaSkeleton(adaMetaSkeleton)
  , mNodeHandle(nodeHandle)
  , mTargetFoodRanker(ranker)
  , mFaceZOffset(faceZOffset)
  , mRemoveRotationForFood(removeRotationForFood)

{
  if (!mNodeHandle)
    throw std::invalid_argument("Ros nodeHandle is nullptr.");
  std::string detectorDataURI
      = getRosParam<std::string>("/perception/detectorDataUri", *mNodeHandle);
  std::string referenceFrameName = getRosParam<std::string>(
      "/perception/referenceFrameName", *mNodeHandle);
  std::string foodDetectorTopicName = getRosParam<std::string>(
      "/perception/foodDetectorTopicName", *mNodeHandle);
  std::string faceDetectorTopicName = getRosParam<std::string>(
      "/perception/faceDetectorTopicName", *mNodeHandle);

  const auto resourceRetriever
      = std::make_shared<aikido::io::CatkinResourceRetriever>();

  mAssetDatabase = std::make_shared<aikido::perception::AssetDatabase>(
      resourceRetriever, detectorDataURI);

  mFoodDetector = std::unique_ptr<aikido::perception::PoseEstimatorModule>(
      new aikido::perception::PoseEstimatorModule(
          *mNodeHandle,
          foodDetectorTopicName,
          mAssetDatabase,
          resourceRetriever,
          referenceFrameName,
          aikido::robot::util::getBodyNodeOrThrow(
              *adaMetaSkeleton, referenceFrameName)));

  mFaceDetector = std::unique_ptr<aikido::perception::PoseEstimatorModule>(
      new aikido::perception::PoseEstimatorModule(
          *mNodeHandle,
          faceDetectorTopicName,
          mAssetDatabase,
          resourceRetriever,
          referenceFrameName,
          aikido::robot::util::getBodyNodeOrThrow(
              *adaMetaSkeleton, referenceFrameName)));

  mPerceptionTimeout
      = getRosParam<double>("/perception/timeoutSeconds", *mNodeHandle);
  mPerceivedFaceName
      = getRosParam<std::string>("/perception/faceName", *mNodeHandle);
  mFoodNames
      = getRosParam<std::vector<std::string>>("/foodItems/names", *nodeHandle);

  if (!mTargetFoodRanker)
    throw std::invalid_argument("TargetFoodRanker not set for perception.");

  // mForkSubscriber = mNodeHandle->subscribe<geometry_msgs::Pose2D>(
  //   "/fork_uv",
  //   1,
  //   boost::bind(&Perception::correctForkTip, this, _1)
  //   );

  mCameraInfoTopic = "/camera/color/camera_info";
  mImageTopic = "/camera/color/image_raw";

  auto joint = mAda->getEndEffectorBodyNode()->getParentJoint();
  if (!joint)
  {
    throw std::runtime_error("Could not find joint");
  }
  mDefaultEETransform
      = Eigen::Isometry3d(joint->getTransformFromParentBodyNode());
}

//==============================================================================
std::vector<std::unique_ptr<FoodItem>> Perception::perceiveFood(
    const std::string& foodName)
{
  if (foodName != ""
      & (std::find(mFoodNames.begin(), mFoodNames.end(), foodName)
         == mFoodNames.end()))
  {
    std::stringstream ss;
    ss << "[" << foodName << "] is unknown." << std::endl;
    throw std::invalid_argument(ss.str());
  }

  std::vector<std::unique_ptr<FoodItem>> detectedFoodItems;

  // Detect items
  std::vector<DetectedObject> detectedObjects;
  mFoodDetector->detectObjects(
      mWorld,
      ros::Duration(mPerceptionTimeout),
      ros::Time(0),
      &detectedObjects);

  std::cout << "Detected " << detectedObjects.size() << " " << foodName
            << std::endl;
  detectedFoodItems.reserve(detectedObjects.size());

  // mCorrectForkTip = true;

  // while(true)
  // {
  //   if (!mCorrectForkTip)
  //     break;

  //   std::cout << "Waiting for correctForkTip " << std::endl;
  //   std::this_thread::sleep_for(std::chrono::seconds(1));
  // }
  Eigen::Isometry3d forqueTF
      = mAda->getEndEffectorBodyNode()
            ->getWorldTransform();

  for (const auto& item : detectedObjects)
  {
    auto foodItem = mTargetFoodRanker->createFoodItem(item, forqueTF);

    if (mRemoveRotationForFood)
    {
      removeRotation(foodItem.get());
      std::cout << "rotation removed added \n";
    }

    std::cout << "A: " << foodItem->getName() << " B: " << foodName << std::endl;
    if (foodName != "" && foodItem->getName() != foodName)
    {
      std::cout << "Not added \n";
      continue;
    }
    detectedFoodItems.emplace_back(std::move(foodItem));
  }

  // sort
  mTargetFoodRanker->sort(detectedFoodItems);
  return detectedFoodItems;
}

//==============================================================================
static Eigen::Isometry3d oldFaceTransform;
static bool saved = false;
Eigen::Isometry3d Perception::perceiveFace()
{
  std::vector<DetectedObject> detectedObjects;

  if (!mFaceDetector->detectObjects(
          mWorld,
          ros::Duration(mPerceptionTimeout),
          ros::Time(0),
          &detectedObjects))
  {
    ROS_WARN("face perception failed");
    throw std::runtime_error("Face perception failed");
  }

  // TODO: the needs to be updated
  // just choose one for now
  for (int skeletonFrameIdx = 0; skeletonFrameIdx < 5; skeletonFrameIdx++)
  {
    auto perceivedFace = mWorld->getSkeleton(
        mPerceivedFaceName + "_" + std::to_string(skeletonFrameIdx));

    if (perceivedFace != nullptr)
    {
      auto faceTransform = perceivedFace->getBodyNode(0)->getWorldTransform();

      // fixed distance:
      double fixedFaceY
          = getRosParam<double>("/feedingDemo/fixedFaceY", *mNodeHandle);
      faceTransform.translation().y() = fixedFaceY;
      faceTransform.translation().z() -= 0.01;

      oldFaceTransform = faceTransform;
      saved = true;
      return faceTransform;
    }
  }
  if (saved)
  {
    return oldFaceTransform;
  }
  ROS_WARN("face perception failed");
  throw std::runtime_error("Face perception failed");
}

//==============================================================================
bool Perception::isMouthOpen()
{
  // return mAssetDatabase->mObjData["faceStatus"].as<bool>();
  // ROS_WARN("Always returning true for isMouthOpen");
  // return true;

  // (1) Detect Face
  std::vector<DetectedObject> detectedObjects;

  if (!mFaceDetector->detectObjects(
          mWorld,
          ros::Duration(mPerceptionTimeout),
          ros::Time(0),
          &detectedObjects))
  {
    ROS_WARN("face perception failed");
    return false;
  }

  // (2) Check if mouth open
  for (auto face : detectedObjects)
  {
    try
    {
      auto yamlNode = face.getYamlNode();
      if (yamlNode["mouth-status"].as<std::string>() == "open")
      {
        return true;
      }
    }
    catch (const YAML::Exception& e)
    {
      ROS_WARN_STREAM(
          "[Perception::isMouthOpen] YAML String Exception: " << e.what()
                                                              << std::endl);
      return false;
    }
  }

  return false;
}

//==============================================================================
void Perception::setFoodItemToTrack(FoodItem* target)
{
  mTargetFoodItem = target;
}

//==============================================================================
FoodItem* Perception::getTargetFoodItem()
{
  return mTargetFoodItem;
}

//==============================================================================
Eigen::Isometry3d Perception::getTrackedFoodItemPose()
{
  if (!mTargetFoodItem)
    throw std::runtime_error("Target item not set.");

  auto detectionResult
      = mFoodDetector->detectObjects(mWorld, ros::Duration(mPerceptionTimeout));

  if (!detectionResult)
    ROS_WARN("Failed to detect new update on the target object.");

  // Pose should've been updated since same metaSkeleton is shared.
  if (mRemoveRotationForFood)
  {
    removeRotation(mTargetFoodItem);
  }
  return mTargetFoodItem->getPose();
}

//==============================================================================
void Perception::removeRotation(const FoodItem* item)
{
  Eigen::Isometry3d foodPose(Eigen::Isometry3d::Identity());
  foodPose.translation() = item->getPose().translation();

  // Downcast Joint to FreeJoint
  dart::dynamics::FreeJoint* freejtptr
      = dynamic_cast<dart::dynamics::FreeJoint*>(
          item->getMetaSkeleton()->getJoint(0));

  if (freejtptr == nullptr)
  {
    dtwarn << "[Perception::removeRotation] Could not cast the joint "
              "of the body to a Free Joint so ignoring the object "
           << item->getName() << std::endl;
    return;
  }
  // Fix the food height
  foodPose.translation()[2] = 0.22;
  freejtptr->setTransform(foodPose);
}

//==============================================================================
void Perception::correctForkTip(const geometry_msgs::Pose2D::ConstPtr& msg)
{
  if (!mCorrectForkTip)
    return;

  std::cout << "correctForkTip " << std::endl;
  receiveCameraInfo();

  cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
  receiveImageMessage(cv_ptr);
  cv::Mat image = cv_ptr->image;

  ROS_INFO("Received Command msg");
  std::cout << msg->x << " " << msg->y << std::endl;

  auto endEffector = mAda->getHand()->getEndEffectorBodyNode();
  std::cout << "tip\n"
            << endEffector->getWorldTransform().translation().transpose()
            << std::endl;

  tf::StampedTransform tfStampedTransform;
  tf::TransformListener tfListener;

  if (!mNodeHandle->ok())
    throw std::runtime_error("Node not ok");

  try
  {
    tfListener.waitForTransform(
        "/camera_color_optical_frame",
        "/map",
        ros::Time(0),
        ros::Duration(10.0));

    tfListener.lookupTransform(
        "/camera_color_optical_frame",
        "/map",
        ros::Time(0),
        tfStampedTransform);
  }
  catch (tf::TransformException ex)
  {
    throw std::runtime_error(
        "Failed to get TF Transform: " + std::string(ex.what()));
  }

  Eigen::Isometry3d map2camera;
  tf::transformTFToEigen(tfStampedTransform, map2camera);

  cv::Mat rmat;
  cv::Mat rvec;
  cv::Mat tvec;

  // get rvec
  Eigen::MatrixXd rot = map2camera.linear();
  cv::eigen2cv(rot, rmat);
  cv::Rodrigues(rmat, rvec);

  // get tvec
  Eigen::Vector3d translation(map2camera.translation());
  cv::eigen2cv(translation, tvec);

  // correct fork tip
  auto joint = mAda->getMetaSkeleton()->getJoint("j2n6s200_joint_forque");

  double minDist(100.0);
  double minXrotation, minYrotation;

  for (double x = -M_PI * 0.5; x < M_PI * 0.5; x += 0.01)
  {
    for (double y = -M_PI * 0.5; y < M_PI * 0.5; y += 0.01)
    {
      // transform of forque
      Eigen::Isometry3d eeTransform(mDefaultEETransform);
      eeTransform.linear() = eeTransform.linear()
                             * Eigen::AngleAxisd(y, Eigen::Vector3d::UnitY())
                             * Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX());

      joint->setTransformFromParentBodyNode(eeTransform);

      Eigen::Vector3d tip = endEffector->getWorldTransform().translation();
      std::vector<cv::Point3f> tip_cv;
      tip_cv.push_back(cv::Point3f(tip[0], tip[1], tip[2]));

      std::vector<cv::Point2f> imagePoints;
      cv::projectPoints(
          tip_cv,
          rvec,
          tvec,
          mCameraModel.intrinsicMatrix(),
          mCameraModel.distortionCoeffs(),
          imagePoints);

      double xdist = msg->x - imagePoints[0].x;
      double ydist = msg->y - imagePoints[0].y;

      double dist = xdist * xdist + ydist * ydist;

      if (dist < minDist)
      {
        minXrotation = x;
        minYrotation = y;
        minDist = dist;
      }
    }
  }

  // transform of forque
  Eigen::Isometry3d eeTransform(mDefaultEETransform);
  eeTransform.linear()
      = eeTransform.linear()
        * Eigen::AngleAxisd(minYrotation, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(minXrotation, Eigen::Vector3d::UnitX());

  std::cout << "Original Fork Transform" << std::endl;
  std::cout << mDefaultEETransform.matrix() << std::endl;

  std::cout << "Corrected " << std::endl;
  std::cout << eeTransform.matrix() << std::endl;
  joint->setTransformFromParentBodyNode(eeTransform);

  Eigen::Vector3d tip = endEffector->getWorldTransform().translation();
  std::vector<cv::Point3f> tip_cv;
  tip_cv.push_back(cv::Point3f(tip[0], tip[1], tip[2]));

  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(
      tip_cv,
      rvec,
      tvec,
      mCameraModel.intrinsicMatrix(),
      mCameraModel.distortionCoeffs(),
      imagePoints);

  // std::cout << "Projected" << std::endl;
  cv::Mat img = image.clone();

  for (auto point : imagePoints)
  {
    // std::cout << point.x << " " << point.y << std::endl;
    cv::circle(img, point, 7, cv::Scalar(255, 255, 255, 255), 5);
  }

  std::stringstream ss;
  ss << "Projected_" << minXrotation << "_" << minYrotation << ".jpg";

  cv::circle(
      img, cv::Point2f(msg->x, msg->y), 3, cv::Scalar(0, 0, 255, 255), 5);
  cv::imwrite(ss.str(), img);
  std::cout << "Save to " << ss.str() << std::endl;

  mCorrectForkTip = false;
}

//=============================================================================
void Perception::receiveCameraInfo()
{
  sensor_msgs::CameraInfoConstPtr info
      = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
          mCameraInfoTopic, *mNodeHandle, ros::Duration(5));
  if (info == nullptr)
  {
    ROS_ERROR("nullptr camera info");
    return;
  }

  mCameraModel.fromCameraInfo(info);
}

//=============================================================================
void Perception::receiveImageMessage(cv_bridge::CvImagePtr& cv_ptr)
{

  sensor_msgs::ImageConstPtr msg
      = ros::topic::waitForMessage<sensor_msgs::Image>(
          mImageTopic, *mNodeHandle, ros::Duration(20));
  if (msg == nullptr)
  {
    ROS_ERROR("nullptr image message");
    return;
  }

  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
}

//=============================================================================
void Perception::setCorrectForkTip(bool val)
{
  mCorrectForkTip = val;
}

} // namespace feeding
