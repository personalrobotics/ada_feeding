#ifndef FEEDING_ACQUISITIONACTION_HPP_
#define FEEDING_ACQUISITIONACTION_HPP_

#include <Eigen/Dense>
#include <ros/ros.h>

namespace feeding {

// Acquisition Action Struct
typedef struct AcquisitionAction {
  // In Food Ref Frame (unless stated)
  // Origin: center of food, Z == table
  // +Z == against gravity == World +Z
  // +X == food major axis

  // Approach Frame
  // Food frame, but +X ==
  //    -(pre_transform.translation())
  //    projected onto X-Y plane

  // Default: Vertical Skewer

  // Utensil Initial Transform in Food Frame
  // Default: Straight above vertical skewer
  Eigen::Isometry3d pre_transform =
      Eigen::Translation3d(Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) *
      Eigen::Isometry3d::Identity();

  // Approach Offset into Food
  Eigen::Vector3d pre_offset = Eigen::Vector3d::Zero();

  // Force Threshold into Food
  double pre_force = 15.0;
  double pre_torque = 4.0;

  // In-Food Twist
  // (dX, dY, dZ) in *approach frame*
  Eigen::Vector3d grasp_offset = Eigen::Vector3d::Zero();
  // (omegaX, omegaY, omegaZ) in *utensil frame*
  Eigen::Vector3d grasp_rot = Eigen::Vector3d::Zero();
  double grasp_duration = 0.0;

  // In-Food Force Threshold
  double grasp_force = 15.0;
  double grasp_torque = 4.0;

  // Extraction Twist
  // (dX, dY, dZ) in *approach frame*
  Eigen::Vector3d ext_offset = Eigen::Vector3d::UnitZ();
  // (omegaX, omegaY, omegaZ) in *utensil frame*
  Eigen::Vector3d ext_rot = Eigen::Vector3d::Zero();
  double ext_duration = 1.0;

  // Extraction Force Threshold
  double ext_force = 50.0;
  double ext_torque = 2.0;

  static std::vector<AcquisitionAction>
  fromRosParam(XmlRpc::XmlRpcValue param) {
    std::vector<AcquisitionAction> ret;
    // From single struct
    if (param.getType() == XmlRpc::XmlRpcValue::Type::TypeStruct) {
      AcquisitionAction defaultAction;
      AcquisitionAction action;

      // Utensil Initial Transform
      if (param.hasMember("pre_pos") &&
          param["pre_pos"].getType() == XmlRpc::XmlRpcValue::Type::TypeArray &&
          param["pre_pos"].size() == 3)
        action.pre_transform.translation() = Eigen::Vector3d(
            param["pre_pos"][0], param["pre_pos"][1], param["pre_pos"][2]);
      if (param.hasMember("pre_quat") &&
          param["pre_quat"].getType() == XmlRpc::XmlRpcValue::Type::TypeArray &&
          param["pre_quat"].size() == 4)
        action.pre_transform.linear() =
            Eigen::Quaterniond(param["pre_quat"][0], param["pre_quat"][1],
                               param["pre_quat"][2], param["pre_quat"][3])
                .toRotationMatrix();

      // Utensil Initial Offset
      if (param.hasMember("pre_offset") &&
          param["pre_offset"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeArray &&
          param["pre_offset"].size() == 3)
        action.pre_offset =
            Eigen::Vector3d(param["pre_offset"][0], param["pre_offset"][1],
                            param["pre_offset"][2]);

      // Pregrasp Force/Torque Threshold
      if (param.hasMember("pre_force") &&
          param["pre_force"].getType() == XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.pre_force = param["pre_force"];
      if (param.hasMember("pre_torque") &&
          param["pre_torque"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.pre_torque = param["pre_torque"];

      // In-Food Twist
      if (param.hasMember("grasp_offset") &&
          param["grasp_offset"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeArray &&
          param["grasp_offset"].size() == 3)
        action.grasp_offset << param["grasp_offset"][0],
            param["grasp_offset"][1], param["grasp_offset"][2];
      if (param.hasMember("grasp_rot") &&
          param["grasp_rot"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeArray &&
          param["grasp_rot"].size() == 3)
        action.grasp_rot << param["grasp_rot"][0], param["grasp_rot"][1],
            param["grasp_rot"][2];
      if (param.hasMember("grasp_duration") &&
          param["grasp_duration"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.grasp_duration = param["grasp_duration"];

      // Grasp Force/Torque Threshold
      if (param.hasMember("grasp_force") &&
          param["grasp_force"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.grasp_force = param["grasp_force"];
      if (param.hasMember("grasp_torque") &&
          param["grasp_torque"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.grasp_torque = param["grasp_torque"];

      // Extraction Twist
      if (param.hasMember("ext_offset") &&
          param["ext_offset"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeArray &&
          param["ext_offset"].size() == 3)
        action.ext_offset << param["ext_offset"][0], param["ext_offset"][1],
            param["ext_offset"][2];
      if (param.hasMember("ext_rot") &&
          param["ext_rot"].getType() == XmlRpc::XmlRpcValue::Type::TypeArray &&
          param["ext_rot"].size() == 3)
        action.ext_rot << param["ext_rot"][0], param["ext_rot"][1],
            param["ext_rot"][2];
      if (param.hasMember("ext_duration") &&
          param["ext_duration"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.ext_duration = param["ext_duration"];

      // Extraction Force/Torque Threshold
      if (param.hasMember("ext_force") &&
          param["ext_force"].getType() == XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.ext_force = param["ext_force"];
      if (param.hasMember("ext_torque") &&
          param["ext_torque"].getType() ==
              XmlRpc::XmlRpcValue::Type::TypeDouble)
        action.ext_torque = param["ext_torque"];

      if (!memcmp(&action, &defaultAction, sizeof(AcquisitionAction))) {
        ROS_WARN("Created default action from ROS Param. Make sure this is "
                 "intentional.");
      }

      ret.push_back(action);
    }
    // From array of structs
    else if (param.getType() == XmlRpc::XmlRpcValue::Type::TypeArray) {
      for (int i = 0; i < param.size(); i++) {
        auto actions = fromRosParam(param[i]);
        ret.insert(std::end(ret), std::begin(actions), std::end(actions));
      }
    } else {
      ROS_WARN("AcquistionAction params must be struct or array!");
    }

    return ret;
  }

} AcquisitionAction;

} // namespace feeding

#endif // FEEDING_ACQUISITIONACTION_HPP_
