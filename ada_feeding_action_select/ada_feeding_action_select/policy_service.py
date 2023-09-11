#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a node that launches a 2 ROS2 services.
This service implement AcquisitionSelect and AcquisitionReport.
"""

# Standard imports
from typing import Dict, Union
import numpy as np

# Third-party imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

# Internal imports
from ada_feeding.helpers import import_from_string
from ada_feeding_perception.helpers import get_img_msg_type, ros_msg_to_cv2_image
from ada_feeding_action_select.policies import Policy
from ada_feeding_action_select.adapters import ContextAdapter, PosthocAdapter
from ada_feeding_msgs.srv import AcquisitionSelect, AcquisitionReport


class PolicyServices(Node):
    """
    The PolicyServices node initializes a 3 things:
    a context adapter converts the incoming mask and images to a context vector
    a policy recommends an action given the context
    a posthoc adapter converts any incoming posthoc data to a posthoc vector.

    This node collects images (rgb + depth) to pass to the context adapter
    and creates services to implement AcquisitionSelect and AcquisitionReport.

    TODO: optionally record data (context + posthoc + loss) based on param
    """

    def __init__(self):
        """
        Declare ROS2 Parameters
        """
        super().__init__("policy_service")

        # Name of the Policy
        policy_param = self.declare_parameter(
            "policy",
            descriptor=ParameterDescriptor(
                name="policy",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "Which policy to use. "
                    "Names must correspond to their own param namespace within this node."
                ),
                read_only=True,
            ),
        )
        policy_name = policy_param.value
        if policy_name is None:
            raise ValueError("Policy Name is required.")

        # Import the policy class
        policy_cls_param = self.declare_parameter(
            f"{policy_name}.policy_class",
            descriptor=ParameterDescriptor(
                name="policy_class",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The class of the policy to run, must subclass "
                    "Policy. E.g., ada_feeding_action_select.policies.ConstantPolicy"
                ),
                read_only=True,
            ),
        )

        policy_cls_name = policy_cls_param.value
        if policy_cls_name is None:
            raise ValueError("Policy Class is required.")

        policy_cls = import_from_string(policy_cls_name)
        assert issubclass(policy_cls, Policy), f"{policy_cls_name} must subclass Policy"

        kwargs = self.get_kwargs(f"{policy_name}.kws", f"{policy_name}.kwargs")

        # Get the context adapter
        context_cls_param = self.declare_parameter(
            "context_class",
            descriptor=ParameterDescriptor(
                name="context_class",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The class of the context adapter to run, must subclass "
                    "ContextAdapter. E.g., ada_feeding_action_select.adapters.NoContext"
                ),
                read_only=True,
            ),
        )

        context_cls_name = context_cls_param.value
        if context_cls_name is None:
            raise ValueError("Context Adapter is required.")

        context_cls = import_from_string(context_cls_name)
        assert issubclass(
            context_cls, ContextAdapter
        ), f"{context_cls_name} must subclass ContextAdapter"

        context_kwargs = self.get_kwargs("context_kws", "context_kwargs")
        self.context_adapter = context_cls(**context_kwargs)

        # Get the posthoc adapter
        posthoc_cls_param = self.declare_parameter(
            "posthoc_class",
            descriptor=ParameterDescriptor(
                name="posthoc_class",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The class of the posthoc adapter to run, must subclass "
                    "PosthocAdapter. E.g., ada_feeding_action_select.adapters.NoContext"
                ),
                read_only=True,
            ),
        )

        posthoc_cls_name = posthoc_cls_param.value
        if posthoc_cls_name is None:
            raise ValueError("Posthoc Adapter is required.")

        posthoc_cls = import_from_string(posthoc_cls_name)
        assert issubclass(
            posthoc_cls, PosthocAdapter
        ), f"{posthoc_cls_name} must subclass PosthocAdapter"

        posthoc_kwargs = self.get_kwargs("posthoc_kws", "posthoc_kwargs")
        self.posthoc_adapter = context_cls(**posthoc_kwargs)

        # Initialize and validate the policy
        self.policy = policy_cls(
            self.context_adapter.dim, self.posthoc_adapter.dim, **kwargs
        )

        # Start image subscribers
        self.image = None
        self.depth = None
        self.ros_objs = []

        if self.context_adapter.need_rgb:
            self.ros_objs.append(self.create_subscription(
                get_img_msg_type("~/image", self), "~/image", self.rgb_callback, 1
            ))
        if self.context_adapter.need_depth:
            self.ros_objs.append(self.create_subscription(
                get_img_msg_type("~/depth", self), "~/depth", self.depth_callback, 1
            ))

        self.ros_objs.append(self.create_service(
            AcquisitionSelect, "~/action_select", self.select_callback
        ))
        self.ros_objs.append(self.create_service(
            AcquisitionReport, "~/action_report", self.report_callback
        ))

        self.get_logger().info(f"Policy '{policy_name}' initialized!")

    # Subscriptions
    def rgb_callback(self, msg: Union[CompressedImage, Image]):
        """
        Store rgb image
        """
        self.image = ros_msg_to_cv2_image(msg)

    def depth_callback(self, msg: Union[CompressedImage, Image]):
        """
        Store depth image
        """
        self.depth = ros_msg_to_cv2_image(msg)

    # Services
    def select_callback(self, request, response):
        """
        Implement AcquisitionSelect.srv
        """

        # Run the context adapter
        context = self.context_adapter.get_context(
            request.food_context, self.image, self.depth
        )
        if context.size != self.context_adapter.dim:
            response.status = "Bad Context"
            return response

        # Run the policy
        response.status = "Success"
        response = self.policy.choice(context, response)
        return response

    def report_callback(self, request, response):
        """
        Implement AcquisitionReport.srv
        """

        # Run the context adapter
        posthoc = self.posthoc_adapter.get_posthoc(np.array(request.posthoc))
        if posthoc.size != self.posthoc_adapter.dim:
            response.status = "Bad Posthoc"
            response.success = False
            return response

        # Run the policy
        response.status = "Success"
        response.success = True
        response = self.policy.update(posthoc, request, response)
        return response

    def get_kwargs(self, kws_root: str, kwarg_root: str) -> Dict:
        """
        Pull variable keyward arguments from ROS2 parameter server.
        Needed because RCL does not allow dictionary params.

        Parameters
        ----------
        kws_root: String array of kwarg keys
        kwarg_root: YAML dictionary of the form "kw: object"

        Returns
        -------
        Dictionary of kwargs: "{kw: object}"
        """

        kws = self.declare_parameter(
            kws_root,
            descriptor=ParameterDescriptor(
                name="kws",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of keywords for custom arguments to be passed "
                    "to the class during initialization."
                ),
                read_only=True,
            ),
        )
        if kws.value is not None:
            kwargs = {
                kw: self.declare_parameter(
                    f"{kwarg_root}.{kw}",
                    descriptor=ParameterDescriptor(
                        name=kw,
                        description="Custom keyword argument for the class.",
                        dynamic_typing=True,
                        read_only=True,
                    ),
                )
                for kw in kws.value
            }
        else:
            kwargs = {}

        return {kw: arg.value for kw, arg in kwargs.items()}


def main():
    """
    Entry point
    """
    rclpy.init()

    node = PolicyServices()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
