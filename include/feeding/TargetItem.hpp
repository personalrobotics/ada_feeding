#ifndef FEEDING_TARGETITEM_HPP_
#define FEEDING_TARGETITEM_HPP_

namespace feeding {

enum TargetItem { FOOD, PLATE, FORQUE, PERSON };

static const std::map<TargetItem, const std::string> TargetToString{
    {FOOD, "food"}, {PLATE, "plate"}, {FORQUE, "forque"}, {PERSON, "person"}};

} // namespace feeding

#endif
