#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script analyzes the output of the holistic_benchmark.py script.
It loads one or more JSON result files, concatenates them, and generates
a series of comparative plots to evaluate the different planning methodologies.

Improvements in this version:
- Centralized constants for plot styling and ordering.
- A helper function to streamline extraction of granular waypoint data.
- New plots for Articutool joint angles, motion smoothness (jerk), and
  a 3D visualization of where tasks succeed or fail in the workspace.
- A text-based summary of results printed to the console.
"""

import argparse
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import List, Dict, Any
import numpy as np

# --- Plotting Constants ---
# Centralize category order and color maps to ensure consistency across all plots.
PLANNING_MODE_ORDER = [
    "joint_unconstrained",
    "task_pos_unconstrained",
    "task_pos_constrained",
    "task_pos_yaw_constrained",
]
STATUS_ORDER = [
    "Success",
    "Path Verification Failure",
    "Planner Failure",
    "Start State Search Failed",
]
STATUS_COLOR_MAP = {
    "Success": "#2ca02c",  # Green
    "Path Verification Failure": "#ff7f0e",  # Orange
    "Planner Failure": "#d62728",  # Red
    "Start State Search Failed": "#9467bd",  # Purple
}
# Preferred joint ranges for manifold adherence, based on prior analysis.
# Jaco Joint 2 (index 1) prefers negative values.
# Jaco Joint 3 (index 2) prefers positive values.
PREFERRED_JOINT_2_RANGE = (-float("inf"), 0.0)
PREFERRED_JOINT_3_RANGE = (0.0, float("inf"))


def load_data(file_paths: List[str]) -> pd.DataFrame:
    """Loads and concatenates data from multiple JSON files."""
    all_data = []
    for file_path in file_paths:
        print(f"Loading data from {file_path}...")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(
                f"Warning: Could not load or parse {file_path}. Error: {e}",
                file=sys.stderr,
            )
    if not all_data:
        print("Error: No data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully loaded {len(all_data)} trials from {len(file_paths)} file(s).")
    return pd.json_normalize(all_data, sep="_")


def _extract_waypoint_data(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to parse and explode waypoint data from the main dataframe."""
    waypoint_rows = []
    for _, row in df.iterrows():
        waypoints = row.get("trajectory_metrics_waypoints_data")
        if isinstance(waypoints, list):
            for wp in waypoints:
                if not isinstance(wp, dict):
                    continue

                # Extract data from the nested dictionaries with checks
                at_sol = wp.get("articutool_solution_rad", {}) or {}
                at_vel = wp.get("articutool_velocities_rad_per_sec", {}) or {}
                ee_pose = wp.get("ee_pose_world", {}) or {}

                waypoint_rows.append(
                    {
                        "planning_mode": row["planning_mode"],
                        "task_id": row["task_id"],
                        "jaco_positions": wp.get("jaco_positions_rad"),
                        "ee_position": ee_pose.get("position"),
                        "time": wp.get("time_from_start_sec", 0.0),
                        "pitch_angle": at_sol.get("pitch"),
                        "roll_angle": at_sol.get("roll"),
                        "pitch_vel": at_vel.get("pitch_vel"),
                        "roll_vel": at_vel.get("roll_vel"),
                    }
                )
    return pd.DataFrame(waypoint_rows).dropna(subset=["jaco_positions", "ee_position"])


def print_summary_table(df: pd.DataFrame):
    """Prints a text-based summary of trial outcomes to the console."""
    print("\n--- Benchmark Summary ---")
    summary = (
        df.groupby("planning_mode")["status"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )
    summary["Total_Attempts"] = df.groupby("planning_mode").size()

    # Ensure all status columns exist for consistent printing
    for status in STATUS_ORDER:
        if status not in summary.columns:
            summary[status] = 0.0

    # Reorder columns for clarity
    summary = summary[["Total_Attempts"] + STATUS_ORDER]

    print(summary.to_string(float_format="%.2f%%"))
    print("-------------------------\n")


def plot_outcomes(df: pd.DataFrame, output_dir: str):
    """Generates a stacked bar chart of trial outcomes."""
    print("Generating trial outcomes plot...")
    status_counts = df.groupby(["planning_mode", "status"]).size().unstack(fill_value=0)
    status_percentages = status_counts.div(status_counts.sum(axis=1), axis=0) * 100
    status_percentages = status_percentages.reset_index()

    fig = px.bar(
        status_percentages,
        x="planning_mode",
        y=STATUS_ORDER,
        title="<b>Breakdown of Trial Outcomes by Planning Methodology</b>",
        labels={"value": "Percentage of Trials (%)"},
        category_orders={"planning_mode": PLANNING_MODE_ORDER},
        color_discrete_map=STATUS_COLOR_MAP,
    )
    fig.update_layout(
        barmode="stack",
        yaxis_title="Percentage of Total Trials (%)",
        xaxis_title="Planning Methodology",
        legend_title_text="Trial Outcome",
    )
    output_path = os.path.join(output_dir, "comparative_outcomes.html")
    fig.write_html(output_path)
    print(f"Saved outcomes plot to {output_path}")


def plot_planning_times(df: pd.DataFrame, output_dir: str):
    """Generates a box plot of planning times for successful trials."""
    print("Generating planning time plot...")
    success_df = df[df["status"] == "Success"]
    if success_df.empty:
        print("Warning: No successful trials found. Skipping planning time plot.")
        return

    fig = px.box(
        success_df,
        x="planning_mode",
        y="planning_time_s",
        color="planning_mode",
        title="<b>Planning Time for Successful Trials</b>",
        labels={"planning_time_s": "Planning Time (s)"},
        category_orders={"planning_mode": PLANNING_MODE_ORDER},
    )
    fig.update_layout(xaxis_title="Planning Methodology")
    output_path = os.path.join(output_dir, "comparative_planning_times.html")
    fig.write_html(output_path)
    print(f"Saved planning time plot to {output_path}")


def plot_path_lengths(df: pd.DataFrame, output_dir: str):
    """Generates a box plot of joint-space path lengths for successful trials."""
    print("Generating path length plot...")
    success_df = df[df["status"] == "Success"]
    if success_df.empty:
        print("Warning: No successful trials found. Skipping path length plot.")
        return

    fig = px.box(
        success_df,
        x="planning_mode",
        y="trajectory_metrics_joint_space_path_length_rad",
        color="planning_mode",
        title="<b>Joint-Space Path Length for Successful Trials</b>",
        labels={"trajectory_metrics_joint_space_path_length_rad": "Path Length (rad)"},
        category_orders={"planning_mode": PLANNING_MODE_ORDER},
    )
    fig.update_layout(xaxis_title="Planning Methodology")
    output_path = os.path.join(output_dir, "comparative_path_lengths.html")
    fig.write_html(output_path)
    print(f"Saved path length plot to {output_path}")


def plot_articutool_metrics(df: pd.DataFrame, output_dir: str):
    """Generates violin plots for Articutool angles and velocities."""
    print("Generating Articutool performance plots (angles and velocities)...")
    success_df = df[df["status"] == "Success"].copy()
    if success_df.empty:
        print("Warning: No successful trials found. Skipping Articutool plots.")
        return

    wp_df = _extract_waypoint_data(success_df)
    if wp_df.empty:
        print("Warning: No valid waypoint data found. Skipping Articutool plots.")
        return

    # Create a 2x2 subplot
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Required Pitch Angle",
            "Required Roll Angle",
            "Required Pitch Velocity",
            "Required Roll Velocity",
        ),
        vertical_spacing=0.15,
    )

    metrics_to_plot = {
        (1, 1): ("pitch_angle", "Angle (rad)"),
        (1, 2): ("roll_angle", "Angle (rad)"),
        (2, 1): ("pitch_vel", "Velocity (rad/s)"),
        (2, 2): ("roll_vel", "Velocity (rad/s)"),
    }

    for (row, col), (metric, y_title) in metrics_to_plot.items():
        sub_fig = px.violin(
            wp_df,
            x="planning_mode",
            y=metric,
            color="planning_mode",
            category_orders={"planning_mode": PLANNING_MODE_ORDER},
        )
        for trace in sub_fig.data:
            fig.add_trace(trace, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)
        fig.update_xaxes(
            title_text="Planning Methodology" if row == 2 else "",
            showticklabels=True if row == 2 else False,
            row=row,
            col=col,
        )

    fig.update_layout(
        height=800,
        title_text="<b>Articutool Performance Metrics During Successful Trajectories</b>",
        showlegend=False,
    )
    output_path = os.path.join(output_dir, "comparative_articutool_performance.html")
    fig.write_html(output_path)
    print(f"Saved Articutool performance plot to {output_path}")


def plot_manifold_adherence(df: pd.DataFrame, output_dir: str):
    """Calculates and plots the 'Manifold Adherence' for each successful trajectory."""
    print("Generating manifold adherence plot...")
    success_df = df[df["status"] == "Success"].copy()
    if success_df.empty:
        print("Warning: No successful trials found. Skipping manifold adherence plot.")
        return

    wp_df = _extract_waypoint_data(success_df)
    if wp_df.empty:
        print(
            "Warning: No valid waypoint data found. Skipping manifold adherence plot."
        )
        return

    # Calculate adherence for each waypoint
    wp_df["is_adherent"] = wp_df.apply(
        lambda row: PREFERRED_JOINT_2_RANGE[0]
        <= row["jaco_positions"][1]
        <= PREFERRED_JOINT_2_RANGE[1]
        and PREFERRED_JOINT_3_RANGE[0]
        <= row["jaco_positions"][2]
        <= PREFERRED_JOINT_3_RANGE[1],
        axis=1,
    )

    # Group by trajectory and calculate percentage
    adherence_df = (
        wp_df.groupby(["planning_mode", "task_id"])["is_adherent"].mean().reset_index()
    )
    adherence_df["adherence_score"] = adherence_df["is_adherent"] * 100

    fig = px.box(
        adherence_df,
        x="planning_mode",
        y="adherence_score",
        color="planning_mode",
        title="<b>Manifold Adherence of Successful Trajectories</b><br><sup>Percentage of waypoints within preferred ranges for Joints 2 & 3</sup>",
        labels={"adherence_score": "Adherence Score (%)"},
        category_orders={"planning_mode": PLANNING_MODE_ORDER},
    )
    fig.update_yaxes(range=[-5, 105])
    fig.update_layout(xaxis_title="Planning Methodology")
    output_path = os.path.join(output_dir, "comparative_manifold_adherence.html")
    fig.write_html(output_path)
    print(f"Saved manifold adherence plot to {output_path}")


def plot_motion_smoothness(df: pd.DataFrame, output_dir: str):
    """Calculates and plots end-effector jerk as a measure of motion smoothness."""
    print("Generating motion smoothness (jerk) plot...")
    success_df = df[df["status"] == "Success"].copy()
    if success_df.empty:
        print("Warning: No successful trials found. Skipping smoothness plot.")
        return

    wp_df = _extract_waypoint_data(success_df)
    if wp_df.empty:
        print("Warning: No valid EE position data found. Skipping smoothness plot.")
        return

    wp_df = wp_df.sort_values(by=["planning_mode", "task_id", "time"])

    # Calculate derivatives (velocity, acceleration, jerk) in a more robust way
    # to avoid the multi-column assignment error with groupby.

    # Group by individual trajectory
    grouped = wp_df.groupby(["planning_mode", "task_id"])

    # Calculate delta_t once for efficiency
    delta_t = grouped["time"].diff()

    # Position columns
    pos_cols = ["vx", "vy", "vz"]
    wp_df[pos_cols] = pd.DataFrame(wp_df["ee_position"].tolist(), index=wp_df.index)

    # Velocity
    vel_cols = ["vel_x", "vel_y", "vel_z"]
    velocity = grouped[pos_cols].diff().div(delta_t, axis=0)
    velocity.columns = vel_cols  # Rename columns to the target names
    wp_df[vel_cols] = velocity

    # Acceleration
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    acceleration = grouped[vel_cols].diff().div(delta_t, axis=0)
    acceleration.columns = acc_cols
    wp_df[acc_cols] = acceleration

    # Jerk
    jerk_cols = ["jerk_x", "jerk_y", "jerk_z"]
    jerk = grouped[acc_cols].diff().div(delta_t, axis=0)
    jerk.columns = jerk_cols
    wp_df[jerk_cols] = jerk

    wp_df["jerk_magnitude"] = np.linalg.norm(wp_df[jerk_cols].fillna(0), axis=1)

    fig = px.box(
        wp_df,
        x="planning_mode",
        y="jerk_magnitude",
        color="planning_mode",
        title="<b>Motion Smoothness: End-Effector Jerk</b>",
        labels={"jerk_magnitude": "Jerk Magnitude (m/s³)"},
        category_orders={"planning_mode": PLANNING_MODE_ORDER},
    )
    # Use a logarithmic scale for jerk as it can have a very large range,
    # and clip the view to a reasonable upper quantile to hide extreme outliers.
    fig.update_yaxes(type="log")
    fig.update_layout(xaxis_title="Planning Methodology")
    output_path = os.path.join(output_dir, "comparative_motion_smoothness.html")
    fig.write_html(output_path)
    print(f"Saved motion smoothness plot to {output_path}")


def plot_workspace_outcomes(df: pd.DataFrame, output_dir: str):
    """Generates a 3D scatter plot of where tasks succeed and fail."""
    print("Generating workspace outcomes plot...")

    task_space_df = df[df["target_position"].notna()].copy()
    if task_space_df.empty:
        print(
            "Warning: No trials with target positions found. Skipping workspace plot."
        )
        return

    task_space_df[["target_x", "target_y", "target_z"]] = pd.DataFrame(
        task_space_df["target_position"].tolist(), index=task_space_df.index
    )

    # --- FIX STARTS HERE ---
    # Manually create subplots because facet_col is not supported for 3D scatter plots
    # in all versions of Plotly.

    # We only care about task-space modes for this plot
    task_modes = [m for m in PLANNING_MODE_ORDER if "task_pos" in m]

    fig = make_subplots(
        rows=1,
        cols=len(task_modes),
        specs=[[{"type": "scene"}] * len(task_modes)],
        subplot_titles=task_modes,
    )

    # Keep track of which legend items have been added to avoid duplicates
    legend_added = set()

    for i, mode in enumerate(task_modes):
        mode_df = task_space_df[task_space_df["planning_mode"] == mode]

        for status in STATUS_ORDER:
            status_df = mode_df[mode_df["status"] == status]
            if status_df.empty:
                continue

            show_legend_for_trace = status not in legend_added
            legend_added.add(status)

            fig.add_trace(
                go.Scatter3d(
                    x=status_df["target_x"],
                    y=status_df["target_y"],
                    z=status_df["target_z"],
                    mode="markers",
                    marker=dict(
                        color=STATUS_COLOR_MAP.get(status, "grey"),
                        size=3,
                        opacity=0.7,
                    ),
                    name=status,
                    legendgroup=status,
                    showlegend=show_legend_for_trace,
                ),
                row=1,
                col=i + 1,
            )

    fig.update_layout(
        height=600,
        title_text="<b>Success and Failure Locations in the Workspace</b>",
        legend_title_text="Trial Outcome",
    )
    # --- FIX ENDS HERE ---

    output_path = os.path.join(output_dir, "comparative_workspace_outcomes.html")
    fig.write_html(output_path)
    print(f"Saved workspace outcomes plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze results from the holistic benchmark."
    )
    parser.add_argument("json_files", nargs="+", help="One or more JSON result files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_analysis_plots",
        help="Directory to save the output HTML plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.json_files)

    # --- Generate Analysis ---
    print_summary_table(df)
    plot_outcomes(df, args.output_dir)
    plot_planning_times(df, args.output_dir)
    plot_path_lengths(df, args.output_dir)
    plot_articutool_metrics(df, args.output_dir)
    plot_manifold_adherence(df, args.output_dir)
    plot_motion_smoothness(df, args.output_dir)
    plot_workspace_outcomes(df, args.output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
