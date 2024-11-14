#!/usr/bin/env python3
"""
This script, intended to be run in sim, is used to benchmark planner performance.

Specifically, it moves the arm to an above plate configuration, and plans to several
realistic "above food" poses it may be asked to plan to. It logs whether the plan
succeeded, and the plan length (joint space).
"""

from threading import Thread
import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import kinova

from transforms3d._gohlketransforms import quaternion_from_euler

from ada_feeding.behaviors.moveit2 import MoveIt2Plan


def main():
    rclpy.init()

    # Create node for this example
    node = Node("ada_planner_benchmark")

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    # Create the conditions
    planners = ["AnytimePathShortening", "RRTConnectkConfigDefault", "RRTstarkConfigDefault"]
    dxs = [0.0, -0.05, 0.05, 0.1]
    dys = [0.0, -0.05, 0.05, 0.1]
    fork_pitches = [0, np.pi/6, -np.pi/6]
    fork_rolls = [np.pi/2, -np.pi/2, 0, np.pi]

    # Create the MoveIt2 object
    callback_group = ReentrantCallbackGroup()
    moveit2 = MoveIt2(
        node=node,
        joint_names=kinova.joint_names(),
        base_link_name=kinova.base_link_name(),
        end_effector_name="forkTip",
        group_name="jaco_arm",
        callback_group=callback_group,
    )
    moveit2.planner_id = planners[0]
    moveit2.allowed_planning_time = 0.5 # secs

    # First, move above plate
    above_plate_config = [
        -2.3149168248766614,
        3.1444595465032634,
        1.7332586075115999,
        -2.3609596843308234,
        4.43936623280362,
        3.06866544924739,
    ]
    node.get_logger().info("Moving to above plate configuration")
    moveit2.move_to_configuration(above_plate_config)
    moveit2.wait_until_executed()
    node.get_logger().info("Moved to above plate configuration")

    # Then, plan to several above food poses
    # In base frame, when facing the robot, +x is to the right, +y is back, +z is up
    base_position = [
        0.26262263022586224,
        -0.2783553055166875,
        0.22773121634396466,
    ]
    # TODO: Also benchmark planning time!
    n_succ = {planner_id : 0 for planner_id in planners}
    n_fail = {planner_id : 0 for planner_id in planners}
    path_lengths = {planner_id : [] for planner_id in planners}
    total_plans = len(dxs) * len(dys) * len(fork_pitches) * len(fork_rolls)
    num_plan = 0
    for dx in dxs: # m
        for dy in dys: # m
            for fork_pitch in fork_pitches:
                for fork_roll in fork_rolls:
                    # Get the above food pose
                    above_food_position = [
                        base_position[0] + dx,
                        base_position[1] + dy,
                        base_position[2],
                    ]
                    w, x, y, z = quaternion_from_euler(fork_pitch, 0, fork_roll, axes="sxyz")
                    above_food_quat_xyzw = [x, y, z, w]
                    
                    # Print progress
                    num_plan += 1
                    node.get_logger().info(f"Planning to above food pose {num_plan}/{total_plans}")
                    node.get_logger().info(
                        f"position: {above_food_position}, quat_xyzw: {above_food_quat_xyzw}"
                    )

                    # Plan to the above food pose
                    for planner_id in planners:
                        moveit2.planner_id = planner_id
                        trajectory = moveit2.plan(
                            position = above_food_position,
                            quat_xyzw = above_food_quat_xyzw,
                        )
                        if trajectory is None: # Plan failed
                            n_fail[planner_id] += 1
                        else: # Plan succeeded
                            n_succ[planner_id] += 1
                            path_len, joints_len = MoveIt2Plan.get_path_len(trajectory)
                            path_lengths[planner_id].append(path_len)

    # Log results
    for planner_id in planners:
        node.get_logger().info(f"{planner_id}: {n_succ[planner_id]} succeeded, {n_fail[planner_id]} failed")
        node.get_logger().info(f"Success rate: {n_succ[planner_id] / (n_succ[planner_id] + n_fail[planner_id])}")
        if n_succ[planner_id] > 0:
            node.get_logger().info(f"Average path length: {np.mean(path_lengths[planner_id])}")
        else:
            node.get_logger().info("No successful plans")

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
