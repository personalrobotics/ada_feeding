#include "feeding/nodes.hpp"
#include <vector>

namespace feeding {

/// Registration Function List
/// Must be referenced this way to avoid static init order fiasco
/// See
/// https://gamedev.stackexchange.com/questions/17746/entity-component-systems-in-c-how-do-i-discover-types-and-construct-component/17759#17759
inline std::vector<RegistrationFn> &getrFns() {
  static std::vector<RegistrationFn> rFns;
  return rFns;
}

void registerNodeFn(RegistrationFn Fn) { getrFns().push_back(Fn); }

void registerNodes(BT::BehaviorTreeFactory &factory, ros::NodeHandle &nh,
                   ada::Ada &robot) {
  for (auto fn : getrFns()) {
    fn(factory, nh, robot);
  }
}

} // end namespace feeding