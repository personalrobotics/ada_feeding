#include "feeding/action/MoveAboveForque.hpp"

#include <pr_tsr/plate.hpp>

#include <libada/util.hpp>

using ada::util::createBwMatrixForTSR;

namespace feeding {
namespace action {

void moveAboveForque(
    const aikido::constraint::dart::CollisionFreePtr& collisionFree,
    FeedingDemo* feedingDemo)
{
  // Load necessary parameters from feedingDemo
  const std::shared_ptr<::ada::Ada>& ada = feedingDemo->getAda();
  double forkHolderAngle = feedingDemo->mForkHolderAngle;
  std::vector<double> forkHolderTranslation = feedingDemo->mForkHolderTranslation;
  double planningTimeout = feedingDemo->mPlanningTimeout;
  int maxNumTrials = feedingDemo->mMaxNumTrials;

  auto aboveForqueTSR = pr_tsr::getDefaultPlateTSR();
  Eigen::Isometry3d forquePose = Eigen::Isometry3d::Identity();
  // y positive is closer to wheelchair
  // z
  // forquePose.translation() = Eigen::Vector3d{0.57, -0.019, 0.012};
  // forquePose.linear() = Eigen::Matrix3d(Eigen::AngleAxisd(0.15,
  // Eigen::Vector3d::UnitX()));
  forquePose.translation() = Eigen::Vector3d{forkHolderTranslation[0],
                                             forkHolderTranslation[1],
                                             forkHolderTranslation[2]};
  forquePose.linear() = Eigen::Matrix3d(
      Eigen::AngleAxisd(forkHolderAngle, Eigen::Vector3d::UnitX()));
  aboveForqueTSR.mT0_w = forquePose;

  aboveForqueTSR.mBw = createBwMatrixForTSR(0.0001, 0.0001, 0.0001, 0, 0, 0);
  aboveForqueTSR.mTw_e.matrix()
      *= ada->getHand()->getEndEffectorTransform("plate")->matrix();

  if (!ada->moveArmToTSR(
          aboveForqueTSR, collisionFree, planningTimeout, maxNumTrials))
    throw std::runtime_error("Trajectory execution failed");
}

} // namespace action
} // namespace feeding
