#ifndef FEEDING_ACTION_SKEWER_HPP_
#define FEEDING_ACTION_SKEWER_HPP_

#include <libada/Ada.hpp>

#include "feeding/FTThresholdHelper.hpp"
#include "feeding/FeedingDemo.hpp"
#include "feeding/Workspace.hpp"
#include "feeding/perception/Perception.hpp"

namespace feeding {
namespace action {

bool skewer(const std::shared_ptr<Perception> &perception,
            const std::string &foodName, const Eigen::Isometry3d &plate,
            const Eigen::Isometry3d &plateEndEffectorTransform,
            FeedingDemo *feedingDemo);
}
} // namespace feeding

#endif
