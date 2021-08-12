#include "feeding/action/MoveAboveFood.hpp"

#include <aikido/constraint/dart/TSR.hpp>
#include <math.h>

#include <libada/util.hpp>

#include "feeding/AcquisitionAction.hpp"
#include "feeding/action/MoveAbove.hpp"
#include "feeding/util.hpp"

using aikido::constraint::dart::TSR;

// Contains motions which are mainly TSR actions
namespace feeding {
namespace action {

bool moveAboveFood(
    std::string foodName,
    const Eigen::Isometry3d& foodTransform,
    float rotateAngle,
    TiltStyle tiltStyle,
    double rotationTolerance,
    FeedingDemo* feedingDemo,
    double* angleGuess)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  double heightAboveFood = feedingDemo->mFoodTSRParameters.at("height");
  double horizontalTolerance = feedingDemo->mFoodTSRParameters.at("horizontalTolerance");
  double verticalTolerance = feedingDemo->mFoodTSRParameters.at("verticalTolerance");
  // NOTE: Although tiltTolerance was originally passed in as a param, it was never used.
  // double tiltTolerance = feedingDemo->mFoodTSRParameters.at("tiltTolerance");

  Eigen::Isometry3d target;
  Eigen::Isometry3d eeTransform
      = *ada->getHand()->getEndEffectorTransform("food");
  Eigen::AngleAxisd rotation
      = Eigen::AngleAxisd(-rotateAngle, Eigen::Vector3d::UnitZ());
  ROS_WARN_STREAM("Rotate Angle: " << rotateAngle);

  // Apply base rotation to food
  Eigen::Vector3d foodVec = foodTransform.rotation() * Eigen::Vector3d::UnitX();
  double baseRotateAngle = atan2(foodVec[1], foodVec[0]);
  if (angleGuess)
  {
    while (abs(baseRotateAngle - *angleGuess) > (M_PI / 2.0))
    {
      baseRotateAngle += (*angleGuess > baseRotateAngle) ? M_PI : (-M_PI);
    }
  }
  ROS_WARN_STREAM("Food Rotate Angle: " << baseRotateAngle);
  Eigen::AngleAxisd baseRotation
      = Eigen::AngleAxisd(baseRotateAngle, Eigen::Vector3d::UnitZ());
  target = removeRotation(foodTransform);
  auto rotationFreeFoodNames = feedingDemo->mRotationFreeFoodNames;
  if (std::find(
          rotationFreeFoodNames.begin(), rotationFreeFoodNames.end(), foodName)
      == rotationFreeFoodNames.end())
  {
    target.linear() = target.linear() * baseRotation;
  }
  target.translation()[2] = feedingDemo->mTableHeight;
  ROS_WARN_STREAM("Food Height: " << target.translation()[2]);

  if (tiltStyle == TiltStyle::NONE)
  {
    eeTransform.linear() = eeTransform.linear() * rotation;
    eeTransform.translation()[2] = heightAboveFood;
  }
  else if (tiltStyle == TiltStyle::VERTICAL)
  {
    eeTransform.linear() = eeTransform.linear() * rotation
                           * Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitX());
    eeTransform.translation()[2] = heightAboveFood;
  }
  else // angled
  {
    eeTransform.linear()
        = eeTransform.linear() * rotation
          * Eigen::AngleAxisd(-M_PI / 8, Eigen::Vector3d::UnitX());
    eeTransform.translation()
        = Eigen::AngleAxisd(
              rotateAngle,
              Eigen::Vector3d::UnitZ()) // Take into account action rotation
          * Eigen::Vector3d{0,
                            -sin(M_PI * 0.25) * heightAboveFood * 0.7,
                            cos(M_PI * 0.25) * heightAboveFood * 0.9};
  }

  auto retval = moveAbove(
      target,
      eeTransform,
      horizontalTolerance,
      verticalTolerance,
      rotationTolerance,
      0.0,
      feedingDemo);
  return retval;
}

} // namespace action
} // namespace feeding
