#ifndef FEEDING_ACTION_MOVEABOVEPLATE_HPP_
#define FEEDING_ACTION_MOVEABOVEPLATE_HPP_

#include <libada/Ada.hpp>

#include "feeding/FeedingDemo.hpp"

namespace feeding {
namespace action {

bool moveAbovePlate(const Eigen::Isometry3d &plate,
                    const Eigen::Isometry3d &plateEndEffectorTransform,
                    FeedingDemo *feedingDemo);

} // namespace action
} // namespace feeding

#endif
