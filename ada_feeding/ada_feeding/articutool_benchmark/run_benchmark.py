#!/usr/bin/env python3
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script is the main entry point for the End-to-End Assistive Feeding Benchmark.
"""

# Standard imports
import argparse
from threading import Thread
import os
from datetime import datetime

# Third-party imports
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from pymoveit2 import MoveIt2

# Local application imports
from feeding_benchmark.benchmark import EndToEndBenchmark
from feeding_benchmark.constants import (
    JOINT_NAMES_JACO,
    JOINT_NAMES_ATOOL,
    JOINT_NAMES_FULL,
    BASE_LINK_JACO,
    BASE_LINK_ATOOL,
    BASE_LINK_FULL,
    END_EFFECTOR_LINK_JACO,
    END_EFFECTOR_LINK_ATOOL,
    END_EFFECTOR_LINK_FULL,
    PLANNING_GROUP_JACO,
    PLANNING_GROUP_ATOOL,
    PLANNING_GROUP_FULL,
    LOGGER,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end feeding benchmark."
    )
    parser.add_argument(
        "--xacro_file",
        type=str,
        default="/home/regulus/ada_ws/src/ada_ros2/ada_moveit/config/ada.urdf.xacro",
        help="Path to the robot URDF/XACRO file.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of full feeding trials to run.",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="Planning timeout in seconds."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="articutool",
        choices=["articutool", "6dof_baseline", "8dof_baseline"],
        help="The execution mode for the benchmark.",
    )
    parser.add_argument(
        "--benchmark_type",
        type=str,
        default="end-to-end",
        choices=["end-to-end", "transport-only"],
        help="The benchmark type to run",
    )
    args = parser.parse_args()

    # 1. Create a unique, timestamped directory for this benchmark run.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(os.getcwd(), "results", f"run_{args.mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Set the ROS_LOG_DIR environment variable BEFORE rclpy.init().
    # This directs all ROS 2 file logging for this process into our unique directory.
    os.environ["ROS_LOG_DIR"] = output_dir

    rclpy.init()
    node = Node("end_to_end_benchmark_node")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Initialize MoveIt2 for the Jaco arm, Articutool, and Full
    if args.mode == "6dof_baseline":
        moveit2_jaco = MoveIt2(
            node=node,
            joint_names=JOINT_NAMES_JACO,
            base_link_name=BASE_LINK_JACO,
            end_effector_name="forkTip",
            group_name=PLANNING_GROUP_JACO,
            callback_group=ReentrantCallbackGroup(),
        )
        moveit2_atool = None
        moveit2_full = None
    else:
        moveit2_jaco = MoveIt2(
            node=node,
            joint_names=JOINT_NAMES_JACO,
            base_link_name=BASE_LINK_JACO,
            end_effector_name=END_EFFECTOR_LINK_JACO,
            group_name=PLANNING_GROUP_JACO,
            callback_group=ReentrantCallbackGroup(),
        )
        moveit2_atool = MoveIt2(
            node=node,
            joint_names=JOINT_NAMES_ATOOL,
            base_link_name=BASE_LINK_ATOOL,
            end_effector_name=END_EFFECTOR_LINK_ATOOL,
            group_name=PLANNING_GROUP_ATOOL,
            callback_group=ReentrantCallbackGroup(),
        )
        moveit2_full = MoveIt2(
            node=node,
            joint_names=JOINT_NAMES_FULL,
            base_link_name=BASE_LINK_FULL,
            end_effector_name=END_EFFECTOR_LINK_FULL,
            group_name=PLANNING_GROUP_FULL,
            callback_group=ReentrantCallbackGroup(),
        )

    benchmark = EndToEndBenchmark(
        node,
        moveit2_jaco,
        moveit2_atool,
        moveit2_full,
        args.xacro_file,
        args.num_trials,
        args.timeout,
        output_dir=output_dir,
        mode=args.mode,
        benchmark_type=args.benchmark_type,
    )

    try:
        benchmark.run()
    except KeyboardInterrupt:
        LOGGER.info("Benchmark interrupted by user.")
    except Exception as e:
        LOGGER.error(f"An unhandled error occurred: {e}")
    finally:
        LOGGER.info("Shutting down.")
        rclpy.shutdown()
        executor_thread.join()


if __name__ == "__main__":
    main()
