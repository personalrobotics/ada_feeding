#ifndef FEEDING_ACTION_SCOOP_HPP_
#define FEEDING_ACTION_SCOOP_HPP_

#include <libada/Ada.hpp>

#include "feeding/Workspace.hpp"

namespace feeding {
namespace action {

void scoop(const std::shared_ptr<ada::Ada>& ada);
}
} // namespace feeding

#endif