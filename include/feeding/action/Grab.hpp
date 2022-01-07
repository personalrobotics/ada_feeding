#ifndef FEEDING_ACTION_GRAB_HPP_
#define FEEDING_ACTION_GRAB_HPP_

#include <libada/Ada.hpp>

#include "feeding/Workspace.hpp"

namespace feeding {
namespace action {

void grabFood(const std::shared_ptr<ada::Ada> &ada,
              const std::shared_ptr<Workspace> &workspace);

void ungrabAndDeleteFood(const std::shared_ptr<ada::Ada> &ada,
                         const std::shared_ptr<Workspace> &workspace);

} // namespace action
} // namespace feeding

#endif