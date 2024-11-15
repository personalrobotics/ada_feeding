#!/usr/bin/env python3
"""
This script, intended to be run in sim, is used to benchmark planner performance.

Specifically, it moves the arm to an above plate configuration, and plans to several
realistic "above food" poses it may be asked to plan to. It logs whether the plan
succeeded, and the plan length (joint space).

Sample usage of this script:
1. Run feeding (so the workplace walls can be used): `ros2 launch ada_feeding ada_feeding_launch.xml use_estop:=false`
2. Run moveit & populate the planning scene: `ros2 launch ada_planning_scene ada_moveit_launch.xml sim:=mock use_rviz:=false`
3. Wait for the planning scene to initialize (see logs)
4. Run this node: `ros2 run ada_feeding planner_benchmark.py ~/path/to/ada_feeding/ada_feeding/data/`
"""

# Standard imports
from collections import defaultdict, namedtuple
from datetime import datetime
import os
import sys
from threading import Thread
from typing import Optional

# Third-party imports
import numpy as np
from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import kinova
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from transforms3d._gohlketransforms import quaternion_from_euler, quaternion_multiply

# Local imports
from ada_feeding.behaviors.moveit2 import MoveIt2Plan


Condition = namedtuple("Condition", ["x", "y", "z", "qx", "qy", "qz", "qw", "frame_id"])

class Benchmark:
    FORKTIP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)

    def __init__(
        self,
        node: Node,
        moveit2: MoveIt2,
        base_position: tuple[float] = (0.26262263022586224, -0.2783553055166875, 0.22773121634396466),
        base_position_frame: str = "j2n6s200_link_base",
        planners: tuple[str] = ("AnytimePathShortening", "RRTConnectkConfigDefault", "RRTstarkConfigDefault"),
        dxs: tuple[float] = (0.0, -0.05, 0.05, 0.1),
        dys: tuple[float] = (0.0, -0.05, 0.05, 0.1),
        fork_pitches: tuple[float] = (0.0, np.pi/8, -np.pi/8),
        fork_rolls: tuple[float] = (0, np.pi/2, np.pi, -np.pi/2),
    ):
        self.node = node
        self.moveit2 = moveit2
        self.base_position = base_position
        self.base_position_frame = base_position_frame
        self.planners = planners
        self.dxs = dxs
        self.dys = dys
        self.fork_pitches = fork_pitches
        self.fork_rolls = fork_rolls
        self.num_conditions = len(dxs) * len(dys) * len(fork_pitches) * len(fork_rolls)

        self.results: dict[Condition, dict[str, tuple[Optional[JointTrajectory], float]]] = defaultdict(dict)

    def conditions(self):
        for dx in self.dxs:
            for dy in self.dys:
                for fork_pitch in self.fork_pitches:
                    for fork_roll in self.fork_rolls:
                        x = self.base_position[0] + dx
                        y = self.base_position[1] + dy
                        z = self.base_position[2]
                        qw, qx, qy, qz = quaternion_multiply(
                            quaternion_from_euler(0, fork_pitch, fork_roll, "sxyz"),
                            self.FORKTIP_QUAT_WXYZ,
                        )
                        yield Condition(x, y, z, qx, qy, qz, qw, self.base_position_frame)

    def run_condition(self, condition: Condition):
        # Get the above food pose
        above_food_position = (condition.x, condition.y, condition.z)
        above_food_quat_xyzw = (condition.qx, condition.qy, condition.qz, condition.qw)

        # Print progress
        self.node.get_logger().info(f"Planning to above food pose")
        self.node.get_logger().info(
            f"position: {above_food_position}, quat_xyzw: {above_food_quat_xyzw}"
        )

        # Plan to the above food pose
        for planner in self.planners:
            trajectory, elapsed_time = self.plan(
                position = above_food_position,
                quat_xyzw = above_food_quat_xyzw,
                planner = planner,
            )
            self.results[condition][planner] = (trajectory, elapsed_time)

    def plan(
        self, 
        position: tuple[float, float, float], 
        quat_xyzw: tuple[float, float, float, float], 
        planner: str,
    ) -> tuple[Optional[JointTrajectory], float]:
        # Plan to the above food pose
        self.moveit2.planner_id = planner
        self.moveit2.set_position_goal(
            position = position,
            frame_id = self.base_position_frame,
            tolerance = 0.001,
        )
        self.moveit2.set_orientation_goal(
            quat_xyzw = quat_xyzw,
            frame_id = self.base_position_frame,
            tolerance = (0.01, 0.01, 0.01),
            parameterization = 1,
        )
        start_time = self.node.get_clock().now()
        trajectory = self.moveit2.plan()
        elapsed_time = (self.node.get_clock().now() - start_time).nanoseconds / 1e9

        return trajectory, elapsed_time

    def run(self):
        i = 0
        for condition in self.conditions():
            self.node.get_logger().info(f"Planning to condition {i}/{self.num_conditions}")
            self.run_condition(condition)
            i += 1

    def to_csv(self, filename: str, above_plate_config: list[float]):
        header = [
            *[f"start {joint_name}" for joint_name in kinova.joint_names()],
            "goal frame",
            "goal x",
            "goal y",
            "goal z",
            "goal qx",
            "goal qy",
            "goal qz",
            "goal qw",
            "planner",
            "elapsed time",
            "success",
            "path length",
            *[f"path length {joint_name}" for joint_name in kinova.joint_names()],
        ]
        with open(filename, "w") as f:
            f.write(",".join(header) + "\n")
            for condition, planner_results in self.results.items():
                for planner, (trajectory, elapsed_time) in planner_results.items():
                    if trajectory is None:
                        success = 0
                        path_len = ''
                        joint_path_lens = {joint_name: '' for joint_name in kinova.joint_names()}
                    else:
                        success = 1
                        path_len, joint_path_lens = MoveIt2Plan.get_path_len(trajectory)
                    f.write(
                        f"{','.join(map(str, above_plate_config))},"
                        f"{condition.frame_id},"
                        f"{condition.x},"
                        f"{condition.y},"
                        f"{condition.z},"
                        f"{condition.qx},"
                        f"{condition.qy},"
                        f"{condition.qz},"
                        f"{condition.qw},"
                        f"{planner},"
                        f"{elapsed_time},"
                        f"{success},"
                        f"{path_len},"
                        f"{','.join(map(str, [joint_path_lens[joint_name] for joint_name in kinova.joint_names()]))}\n"
                    )

    @staticmethod
    def summary_stats(data: list[float]) -> str:
        return (
            f"mean {np.mean(data)}, "
            f"median {np.median(data)}, "
            f"min {np.min(data)}, "
            f"max {np.max(data)}, "
            f"25th {np.percentile(data, 25)}, "
            f"50th {np.percentile(data, 50)}, "
            f"75th {np.percentile(data, 75)}, "
            f"std {np.std(data)}"
        )

    def log_results(
        self, 
        max_path_len: float = 10.0,
        max_path_len_joint: dict[str, float] = {
            "j2n6s200_joint_1": np.pi * 5.0 / 6.0,
            "j2n6s200_joint_2": np.pi / 2.0,
        },
    ):
        """
        Log the results of the benchmark. Note that max_path_len and max_path_len_joint
        should match what is used in the acquire_food_tree to move above food.

        This method gets the following values for each planner:
        - Number of successful plans
        - Number of failed plans
        - Success rate (successful plans / total plans)
        - Summary statistics of path lengths for successful plans
        - Summary statistics of per-joint path lengths for successful plans
        - Proportion of total plans with path lengths within the maxes
        - Summary statistics for planning time

        Summary statistics report mean, median, min, max, 25th, 50th, 75th percentiles, and standard deviation
        """
        # Get the results
        n_succ = {planner_id : 0 for planner_id in self.planners}
        n_fail = {planner_id : 0 for planner_id in self.planners}
        path_lengths = {planner_id : [] for planner_id in self.planners}
        path_lengths_joint = {planner_id : {joint_name: [] for joint_name in kinova.joint_names()} for planner_id in self.planners}
        n_succ_path_lens = {planner_id : 0 for planner_id in self.planners}
        elapsed_times = {planner_id : [] for planner_id in self.planners}
        for condition, planner_results in self.results.items():
            for planner, (trajectory, elapsed_time) in planner_results.items():
                elapsed_times[planner].append(elapsed_time)
                if trajectory is None:
                    n_fail[planner] += 1
                else:
                    n_succ[planner] += 1
                    path_len, joints_len = MoveIt2Plan.get_path_len(trajectory)
                    path_lengths[planner].append(path_len)
                    satisfied_path_len = path_len <= max_path_len
                    for joint_name, joint_len in joints_len.items():
                        path_lengths_joint[planner][joint_name].append(joint_len)
                        if joint_name in max_path_len_joint:
                            satisfied_path_len = satisfied_path_len and joint_len <= max_path_len_joint[joint_name]
                    if satisfied_path_len:
                        n_succ_path_lens[planner] += 1

        # Log results
        for planner_id in self.planners:
            total_plans = n_succ[planner_id] + n_fail[planner_id]
            self.node.get_logger().info(f"*** {planner_id} ***")
            self.node.get_logger().info(f"    {n_succ[planner_id]} succeeded, {n_fail[planner_id]} failed")
            self.node.get_logger().info(f"    Success rate: {n_succ[planner_id] / total_plans}")
            if n_succ[planner_id] > 0:
                self.node.get_logger().info(f"    Path Lengths: {Benchmark.summary_stats(path_lengths[planner_id])}")
                for joint_name in kinova.joint_names():
                    self.node.get_logger().info(
                        f"        {joint_name} Path Lengths: {Benchmark.summary_stats(path_lengths_joint[planner_id][joint_name])}"
                    )
                self.node.get_logger().info(
                    f"    Proportion of plans with path lengths within maxes: {n_succ_path_lens[planner_id] / total_plans}"
                )
            self.node.get_logger().info(f"    Elapsed Times: {Benchmark.summary_stats(elapsed_times[planner_id])}")

def main(out_dir: Optional[str]):
    rclpy.init()

    # Create node for this example
    node = Node("ada_planner_benchmark")

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

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

    # Create the benchmark object
    benchmark = Benchmark(node, moveit2)

    # Configure the MoveIt2 object
    moveit2.planner_id = benchmark.planners[0]
    moveit2.allowed_planning_time = 5.0 # secs
    moveit2.max_velocity = 1.0
    moveit2.max_acceleration = 0.8

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

    # Run the benchmark
    benchmark.run()

    # Save results
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "planner_benchmark.csv"
        benchmark.to_csv(os.path.join(out_dir, filename), above_plate_config)

    # Log the results
    benchmark.log_results()

    # Cleanup
    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    out_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else None
    main(out_dir)
