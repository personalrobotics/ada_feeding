#include "feeding/nodes.hpp"

#include <vector>

namespace feeding {

/// Registration Function List
static std::vector<RegistrationFn> rFns;

void registerNodeFn(RegistrationFn Fn) { rFns.push_back(Fn); }

void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                   ada::Ada &robot) {
  for (auto fn : rFns) {
    fn(factory, nh, robot);
  }
}

} // end namespace feeding