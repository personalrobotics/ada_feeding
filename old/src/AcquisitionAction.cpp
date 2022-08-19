#include "feeding/AcquisitionAction.hpp"

namespace feeding {

//==============================================================================
AcquisitionAction::AcquisitionAction(TiltStyle tiltStyle, double rotationAngle,
                                     double tiltAngle,
                                     Eigen::Vector3d moveIntoDirection)
    : mTiltStyle(tiltStyle), mRotationAngle(rotationAngle),
      mTiltAngle(tiltAngle), mMoveIntoDirection(moveIntoDirection) {
  // Do nothing
}

//==============================================================================
TiltStyle AcquisitionAction::getTiltStyle() const { return mTiltStyle; }

//==============================================================================
double AcquisitionAction::getRotationAngle() const { return mRotationAngle; }

//==============================================================================
double AcquisitionAction::getTiltAngle() const { return mTiltAngle; }

//==============================================================================
Eigen::Vector3d AcquisitionAction::moveIntoDirection() const {
  return mMoveIntoDirection;
}

} // namespace feeding
