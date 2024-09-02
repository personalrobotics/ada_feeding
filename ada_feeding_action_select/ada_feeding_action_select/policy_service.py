#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a node that launches a 2 ROS2 services.
This service implement AcquisitionSelect and AcquisitionReport.
"""

# Standard imports
import argparse
import copy
import errno
import os
import threading
import time
from typing import Dict
import uuid

# Third-party imports
from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import torch

# Internal imports
from ada_feeding.helpers import import_from_string
from ada_feeding_action_select.helpers import register_logger
from ada_feeding_action_select.policies import Policy
from ada_feeding_action_select.adapters import ContextAdapter, PosthocAdapter
from ada_feeding_msgs.srv import AcquisitionSelect, AcquisitionReport


class PolicyServices(Node):
    """
    The PolicyServices node initializes a 3 things:
    a context adapter converts the incoming mask to a context vector
    a policy recommends an action given the context
    a posthoc adapter converts any incoming posthoc data to a posthoc vector

    TODO: optionally record data (context + posthoc + loss) based on param
    """

    # pylint: disable=too-many-instance-attributes
    # Having more than 7 instance attributes is unavoidable here, since for
    # every subscription we need to store the subscription, mutex, and data,
    # and we have 2 subscriptions.

    def _declare_parameters(self):
        """
        Declare all ROS2 Parameters
        """

        self.declare_parameter(
            name="policy",
            value=None,
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
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    f"{self.get_parameter('policy').value}.policy_class",
                    None,
                    ParameterDescriptor(
                        name=f"{self.get_parameter('policy').value}.policy_class",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The class of the policy to run, must subclass "
                            "Policy. E.g., ada_feeding_action_select.policies.ConstantPolicy"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "context_class",
                    "ada_feeding_action_select.adapters.NoContext",
                    ParameterDescriptor(
                        name="context_class",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The class of the context adapter to run, must subclass "
                            "ContextAdapter. E.g., ada_feeding_action_select.adapters.NoContext"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "posthoc_class",
                    "ada_feeding_action_select.adapters.NoContext",
                    ParameterDescriptor(
                        name="posthoc_class",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The class of the posthoc adapter to run, must subclass "
                            "PosthocAdapter. E.g., ada_feeding_action_select.adapters.NoContext"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "record_dir",
                    None,
                    ParameterDescriptor(
                        name="record_dir",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "Directory to save/load data relative to share/data; "
                            "Files will be saved as <record_dir>/<timestamp>_record.pt; "
                            "Each file contains one AcquisitionSelect and one AcquisitionReport. "
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "checkpoint_dir",
                    None,
                    ParameterDescriptor(
                        name="checkpoint_dir",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "Directory to save/load checkpoints relative to share/data. "
                            "Files will be saved as <checkpoint_dir>/<timestamp>_ckpt.pt"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "checkpoint_save_period",
                    0,
                    ParameterDescriptor(
                        name="checkpoint_save_period",
                        type=ParameterType.PARAMETER_INTEGER,
                        description=(
                            "How often to save a new checkpoint. "
                            "Undefine or set <1 to disable checkpoint saving"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "checkpoint_load_latest",
                    False,
                    ParameterDescriptor(
                        name="checkpoint_load_latest",
                        type=ParameterType.PARAMETER_BOOL,
                        description=(
                            "Whether to load latest checkpoint in dir at start"
                        ),
                        read_only=True,
                    ),
                ),
            ],
        )

    def _init_checkpoints_record(self, context_cls: type, posthoc_cls: type) -> None:
        """
        Seperate logic for checkpoint and data record
        """

        # pylint: disable=too-many-branches

        ### Data Record Initialization
        self.record_dir = None
        if self.get_parameter("record_dir").value is not None:
            self.record_dir = os.path.join(
                get_package_share_directory("ada_feeding_action_select"),
                "data",
                self.get_parameter("record_dir").value,
            )
            if not os.path.isdir(self.record_dir):
                self.get_logger().error(
                    f"Data record dir not found, cannot save: {self.record_dir}"
                )
                self.record_dir = None
        if self.record_dir is not None:
            self.get_logger().info(f"Saving records to: {self.record_dir}")

        ### Checkpoint Initialization
        # Checkpoint Directory
        self.checkpoint_dir = None
        if self.get_parameter("checkpoint_dir").value is not None:
            self.checkpoint_dir = os.path.join(
                get_package_share_directory("ada_feeding_action_select"),
                "data",
                self.get_parameter("checkpoint_dir").value,
            )
            if not os.path.isdir(self.checkpoint_dir):
                self.get_logger().error(
                    f"Checkpoint dir not found, cannot save or load: {self.checkpoint_dir}"
                )
                self.checkpoint_dir = None

        # Checkpoint Saving
        self.checkpoint_save_period = (
            0
            if (self.get_parameter("checkpoint_save_period").value < 1)
            else self.get_parameter("checkpoint_save_period").value
        )
        self.n_successful_reports = 0
        if self.checkpoint_save_period > 0 and self.checkpoint_dir is not None:
            self.get_logger().info(f"Saving checkpoints to: {self.checkpoint_dir}")

        # Load latest checkpoint if requested
        if (
            self.get_parameter("checkpoint_load_latest").value
            and self.checkpoint_dir is not None
        ):
            pt_files = sorted(
                [
                    f
                    for f in os.listdir(self.checkpoint_dir)
                    if os.path.isfile(f) and f.endswith(".pt")
                ]
            )
            if len(pt_files) > 0:
                with open(pt_files[-1], "rb") as ckpt_file:
                    ckpt = torch.load(ckpt_file)
                    try:
                        if ckpt["context_cls"] != context_cls:
                            self.get_logger().warning(
                                f"Context adapter mismatch in checkpoint: {ckpt['context_cls']} != {context_cls}"
                            )
                        if ckpt["posthoc_cls"] != posthoc_cls:
                            self.get_logger().warning(
                                f"Posthoc adapter mismatch in checkpoint: {ckpt['posthoc_cls']} != {posthoc_cls}"
                            )
                        if self.policy.set_checkpoint(ckpt["checkpoint"]):
                            self.get_logger().info(f"Loaded Checkpoint: {pt_files[-1]}")
                        else:
                            self.get_logger().warning(
                                f"Could not set policy checkpoint: {pt_files[-1]}"
                            )
                    except KeyError as error:
                        self.get_logger().warning(f"Malformed checkpoint: {error}")

    def __init__(self):
        """
        Declare ROS2 Parameters
        """
        super().__init__("policy_service")
        register_logger(self.get_logger())
        self._declare_parameters()

        # Name of the Policy
        policy_name = self.get_parameter("policy").value
        if policy_name is None:
            raise ValueError("Policy Name is required.")

        # Import the policy class
        policy_cls_name = self.get_parameter(f"{policy_name}.policy_class").value
        if policy_cls_name is None:
            raise ValueError("Policy Class is required.")

        policy_cls = import_from_string(policy_cls_name)
        assert issubclass(policy_cls, Policy), f"{policy_cls_name} must subclass Policy"

        policy_kwargs = self.get_kwargs(f"{policy_name}.kws", f"{policy_name}.kwargs")

        # Get the context adapter
        context_cls = import_from_string(self.get_parameter("context_class").value)
        assert issubclass(
            context_cls, ContextAdapter
        ), f"{self.get_parameter('context_class').value} must subclass ContextAdapter"

        context_kwargs = self.get_kwargs("context_kws", "context_kwargs")
        self.context_adapter = context_cls(**context_kwargs)

        # Get the posthoc adapter
        posthoc_cls = import_from_string(self.get_parameter("posthoc_class").value)
        assert issubclass(
            posthoc_cls, PosthocAdapter
        ), f"{self.get_parameter('posthoc_class').value} must subclass PosthocAdapter"

        posthoc_kwargs = self.get_kwargs("posthoc_kws", "posthoc_kwargs")
        self.posthoc_adapter = posthoc_cls(**posthoc_kwargs)

        # Initialize and validate the policy
        self.policy = policy_cls(
            self.context_adapter.dim, self.posthoc_adapter.dim, **policy_kwargs
        )

        # Create AcquisitionSelect cache
        # UUID -> {context, request, response}
        self.cache_lock = threading.Lock()
        self.cache = {}

        # Init Checkpoints / Data Record
        self._init_checkpoints_record(context_cls, posthoc_cls)

        # Start ROS services
        self.acquisition_report_threads = []
        self.ros_objs = []
        self.ros_objs.append(
            self.create_service(
                AcquisitionSelect, "~/action_select", self.select_callback
            )
        )
        self.ros_objs.append(
            self.create_service(
                AcquisitionReport, "~/action_report", self.report_callback
            )
        )

        self.get_logger().info(f"Policy '{policy_name}' initialized!")

    # Services
    def select_callback(
        self, request: AcquisitionSelect.Request, response: AcquisitionSelect.Response
    ) -> AcquisitionSelect.Response:
        """
        Implement AcquisitionSelect.srv
        """

        # Run the context adapter
        context = self.context_adapter.get_context(request.food_context)
        if context.size != self.context_adapter.dim:
            response.status = "Bad Context"
            self.get_logger().warning(
                "Context Adapter Failure: Incorrect Context Dim Returned"
            )
            return response

        # Run the policy
        response.status = "Success"
        choice = self.policy.choice(context)
        if isinstance(choice, str):
            response.status = choice
        else:
            res = list(zip(*choice))
            response.probabilities = list(res[0])
            response.actions = list(res[1])
            select_id = str(uuid.uuid4())
            with self.cache_lock:
                self.cache[select_id] = {
                    "context": np.copy(context),
                    "request": copy.deepcopy(request),
                    "response": copy.deepcopy(response),
                }
            response.id = select_id

        if response.status != "Success":
            self.get_logger().warning(f"Policy Choice Failure: '{response.status}'")
        else:
            self.get_logger().info(
                f"AcquisitionSelect Success! Sending Response with ID: '{response.id}'"
            )

        return response

    def report_callback(
        self, request: AcquisitionReport.Request, response: AcquisitionReport.Response
    ) -> AcquisitionReport.Response:
        """
        Implement AcquisitionReport.srv
        """
        self.get_logger().info(
            f"AcquisitionReport Request with ID: '{request.id}' and loss '{request.loss}'"
        )

        # Remove any completed threads
        i = 0
        while i < len(self.acquisition_report_threads):
            if not self.acquisition_report_threads[i].is_alive():
                self.get_logger().info("Removing completed acquisition report thread")
                self.acquisition_report_threads.pop(i)
            else:
                i += 1

        # Start the asynch thread
        request_copy = copy.deepcopy(request)
        response_copy = copy.deepcopy(response)
        # self.report_callback_work(request_copy, response_copy)
        thread = threading.Thread(
            target=self.report_callback_work, args=(request_copy, response_copy)
        )
        self.acquisition_report_threads.append(thread)
        self.get_logger().info("Starting new acquisition report thread")
        thread.start()

        # Return success immediately
        response.status = "Success"
        response.success = True
        return response

    # pylint: disable=too-many-statements
    # One over is fine for this function.
    def report_callback_work(
        self, request: AcquisitionReport.Request, response: AcquisitionReport.Response
    ) -> AcquisitionReport.Response:
        """
        Perform the work of updating the policy based on the acquisition. This is a workaround
        to the fact that either ROSLib or rosbridge (likely the latter) cannot process a service
        and action at the same time, so in practice the next motion waits until after the policy
        has been updated, which adds a few seconds of unnecessary latency.
        """
        with self.cache_lock:
            # Collect cached context
            if request.id not in self.cache:
                response.status = "id does not map to previous select call"
                self.get_logger().error(f"AcquistionReport: {response.status}")
                response.success = False
                return response
            cache = copy.deepcopy(self.cache[request.id])
        context = cache["context"]

        # Collect executed action
        select = cache["response"]
        if request.action_index >= len(select.actions):
            response.status = "action_index out of range"
            self.get_logger().error(f"AcquistionReport: {response.status}")
            response.success = False
            return response
        action = (
            select.probabilities[request.action_index]
            if len(select.probabilities) > request.action_index
            else 1.0,
            select.actions[request.action_index],
        )

        self.get_logger().info(f"Executed Action: '{request.action_index}'")

        # Run the posthoc adapter
        posthoc = self.posthoc_adapter.get_posthoc(np.array(request.posthoc))
        if posthoc.size != self.posthoc_adapter.dim:
            response.status = "Bad Posthoc"
            response.success = False
            return response

        # Run the policy
        response.status = "Success"
        response.success = True
        update = self.policy.update(posthoc, context, action, request.loss)
        if not update[0]:
            response.status = update[1]
            self.get_logger().error(f"AcquistionReport: {response.status}")
            response.success = False
            return response

        # Report completed
        self.n_successful_reports += 1
        with self.cache_lock:
            if request.id in self.cache:
                del self.cache[request.id]

        # Save checkpoint if requested
        if (
            self.checkpoint_save_period > 0
            and self.n_successful_reports % self.checkpoint_save_period == 0
            and self.checkpoint_dir is not None
        ):
            # Save Checkpoint
            filename = f"{time.time()}_ckpt.pt"
            with open(os.path.join(self.checkpoint_dir, filename), "wb") as ckpt_file:
                ckpt = {}
                ckpt["context_cls"] = self.context_adapter.__class__
                ckpt["posthoc_cls"] = self.posthoc_adapter.__class__
                ckpt["checkpoint"] = self.policy.get_checkpoint()
                torch.save(ckpt, ckpt_file)

        # Save data if requested
        if self.record_dir is not None:
            filename = f"{time.time()}_record.pt"
            with open(os.path.join(self.record_dir, filename), "wb") as record_file:
                record = {}
                record["select.request"] = cache["request"]
                record["select.response"] = cache["response"]
                record["report.request"] = copy.deepcopy(request)
                record["report.response"] = copy.deepcopy(response)
                record["context_cls"] = self.context_adapter.__class__
                record["posthoc_cls"] = self.posthoc_adapter.__class__
                record["context"] = cache["context"]
                record["posthoc"] = np.copy(posthoc)
                torch.save(record, record_file)

        return response

    # TODO: Consider making get_kwargs an ada_feeding helper
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


def set_data_folder():
    """
    Entry Point
    Create symlink from shared data directory
    """
    logger = rclpy.logging.get_logger("policy_service")

    parser = argparse.ArgumentParser(
        prog="set_data_folder", description="Set data directory root."
    )
    parser.add_argument(
        "directory", metavar="<directory>", type=os.path.abspath, nargs=1
    )
    args = parser.parse_args()
    data_dir = args.directory[0]
    if not os.path.isdir(data_dir):
        logger.error(f"Error: Not a directory or does not exist; {data_dir}")
        return 1

    link_name = os.path.join(
        get_package_share_directory("ada_feeding_action_select"), "data"
    )
    try:
        os.symlink(data_dir, link_name)
    except OSError as error:
        if error.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(data_dir, link_name)
        else:
            raise error

    logger.info("Success: Set installed data directory.")
    return 0


def main():
    """
    Entry point
    """

    # Check Data Directory Exists
    data_dir = os.path.join(
        get_package_share_directory("ada_feeding_action_select"), "data"
    )
    if not os.path.isdir(data_dir):
        logger = rclpy.logging.get_logger("policy_service")
        logger.error("Error: No data directory set.")
        logger.error(
            "Create a folder where you want checkpoints and records saved in. "
            "Then, symlink that folder to share/data by running: "
            "`ros2 run ada_feeding_action_select set_data_folder <directory>`."
        )
        logger.error("For now, checkpoints and records will not be saved.")

    # Node Setup
    rclpy.init()

    node = PolicyServices()

    rclpy.spin(node)

    rclpy.shutdown()

    return 0


if __name__ == "__main__":
    main()
