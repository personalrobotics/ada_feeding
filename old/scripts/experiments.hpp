#ifndef FEEDING_SCRIPTS_HPP_
#define FEEDING_SCRIPTS_HPP_

#include "feeding/FTThresholdHelper.hpp"
#include "feeding/FeedingDemo.hpp"
#include "feeding/perception/Perception.hpp"
#include "feeding/util.hpp"

/// List of executable scripts supported by main.cpp
namespace feeding {

void acquisition(
    FeedingDemo& feedingDemo,
    ros::NodeHandle nodeHandle);

void bite_location_detector(
    FeedingDemo& feedingDemo,
    FTThresholdHelper& ftThresholdHelper,
    Perception& perception,
    ros::NodeHandle nodeHandle,
    bool autoContinueDemo,
    bool adaReal);

void demo(
    FeedingDemo& feedingDemo,
    std::shared_ptr<Perception>& perception,
    ros::NodeHandle nodeHandle);

void humanStudyDemo(
    FeedingDemo& feedingDemo,
    std::shared_ptr<Perception>& perception,
    std::shared_ptr<ros::NodeHandle> nodeHandle);

void spanetDemo(
    FeedingDemo& feedingDemo,
    std::shared_ptr<Perception>& perception,
    ros::NodeHandle nodeHandle);
};

#endif
