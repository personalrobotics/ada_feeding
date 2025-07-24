#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script analyzes the output of the holistic_benchmark.py script.
It loads one or more JSON result files, concatenates them, and generates
a series of comparative plots to evaluate the different planning methodologies,
including a new "Manifold Adherence" metric.
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

    print(
        f"Successfully loaded a total of {len(all_data)} trials from {len(file_paths)} file(s)."
    )
    return pd.json_normalize(all_data, sep="_")


def plot_outcomes(df: pd.DataFrame, output_dir: str):
    """
    Generates a stacked bar chart showing the proportion of each trial status
    for each planning methodology.
    """
    print("Generating trial outcomes plot...")

    # --- Data Preparation ---
    # Calculate the counts of each status for each planning mode
    status_counts = df.groupby(["planning_mode", "status"]).size().unstack(fill_value=0)

    # Calculate the total attempts for each mode to normalize
    total_attempts = status_counts.sum(axis=1)

    # Calculate percentages
    status_percentages = status_counts.div(total_attempts, axis=0) * 100
    status_percentages = status_percentages.reset_index()

    # Define the desired order of statuses for plotting
    status_order = [
        "Success",
        "Path Verification Failure",
        "Planner Failure",
        "Start State Search Failed",
    ]

    # Ensure all status columns exist, adding any that are missing with a value of 0
    for status in status_order:
        if status not in status_percentages.columns:
            status_percentages[status] = 0

    # --- Plotting ---
    fig = px.bar(
        status_percentages,
        x="planning_mode",
        y=status_order,  # Plot each status as a segment of the bar
        title="<b>Breakdown of Trial Outcomes by Planning Methodology</b>",
        labels={
            "planning_mode": "Planning Methodology",
            "value": "Percentage of Trials (%)",
        },
        category_orders={
            "planning_mode": [
                "joint_unconstrained",
                "task_pos_unconstrained",
                "task_pos_constrained",
                "task_pos_yaw_constrained",
            ]
        },
        color_discrete_map={
            "Success": "#2ca02c",
            "Path Verification Failure": "#ff7f0e",
            "Planner Failure": "#d62728",
            "Start State Search Failed": "#9467bd",
        },
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

    # Filter for successful trials only
    success_df = df[df["status"] == "Success"].copy()

    if success_df.empty:
        print("Warning: No successful trials found. Skipping planning time plot.")
        return

    fig = px.box(
        success_df,
        x="planning_mode",
        y="planning_time_s",
        color="planning_mode",
        title="<b>Planning Time for Successful Trials</b>",
        labels={
            "planning_mode": "Planning Methodology",
            "planning_time_s": "Planning Time (s)",
        },
        category_orders={
            "planning_mode": [
                "joint_unconstrained",
                "task_pos_unconstrained",
                "task_pos_constrained",
                "task_pos_yaw_constrained",
            ]
        },
    )

    output_path = os.path.join(output_dir, "comparative_planning_times.html")
    fig.write_html(output_path)
    print(f"Saved planning time plot to {output_path}")


def plot_path_lengths(df: pd.DataFrame, output_dir: str):
    """Generates a box plot of joint-space path lengths for successful trials."""
    print("Generating path length plot...")

    success_df = df[df["status"] == "Success"].copy()

    if success_df.empty:
        print("Warning: No successful trials found. Skipping path length plot.")
        return

    # Ensure the path length column exists
    if "trajectory_metrics_joint_space_path_length_rad" not in success_df.columns:
        print(
            "Warning: 'trajectory_metrics_joint_space_path_length_rad' column not found. Skipping path length plot."
        )
        return

    fig = px.box(
        success_df,
        x="planning_mode",
        y="trajectory_metrics_joint_space_path_length_rad",
        color="planning_mode",
        title="<b>Joint-Space Path Length for Successful Trials</b>",
        labels={
            "planning_mode": "Planning Methodology",
            "trajectory_metrics_joint_space_path_length_rad": "Path Length (rad)",
        },
        category_orders={
            "planning_mode": [
                "joint_unconstrained",
                "task_pos_unconstrained",
                "task_pos_constrained",
                "task_pos_yaw_constrained",
            ]
        },
    )

    output_path = os.path.join(output_dir, "comparative_path_lengths.html")
    fig.write_html(output_path)
    print(f"Saved path length plot to {output_path}")


def plot_articutool_velocities(df: pd.DataFrame, output_dir: str):
    """Generates violin plots of required Articutool velocities."""
    print("Generating Articutool velocity plots...")

    success_df = df[df["status"] == "Success"].copy()

    if success_df.empty:
        print("Warning: No successful trials found. Skipping Articutool velocity plot.")
        return

    # Explode the waypoints data
    waypoints_data = []
    for _, row in success_df.iterrows():
        if isinstance(row.get("trajectory_metrics_waypoints_data"), list):
            for waypoint in row["trajectory_metrics_waypoints_data"]:
                if waypoint and isinstance(
                    waypoint.get("articutool_velocities_rad_per_sec"), dict
                ):
                    waypoints_data.append(
                        {
                            "planning_mode": row["planning_mode"],
                            "pitch_vel": waypoint[
                                "articutool_velocities_rad_per_sec"
                            ].get("pitch_vel"),
                            "roll_vel": waypoint[
                                "articutool_velocities_rad_per_sec"
                            ].get("roll_vel"),
                        }
                    )

    if not waypoints_data:
        print("Warning: No valid Articutool velocity data found. Skipping plot.")
        return

    vel_df = pd.DataFrame(waypoints_data).dropna()

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Pitch Velocity", "Roll Velocity")
    )

    for i, vel_type in enumerate(["pitch_vel", "roll_vel"]):
        sub_fig = px.violin(
            vel_df,
            x="planning_mode",
            y=vel_type,
            color="planning_mode",
            category_orders={
                "planning_mode": [
                    "joint_unconstrained",
                    "task_pos_unconstrained",
                    "task_pos_constrained",
                    "task_pos_yaw_constrained",
                ]
            },
        )
        for trace in sub_fig.data:
            fig.add_trace(trace, row=1, col=i + 1)

    fig.update_layout(
        title_text="<b>Required Articutool Velocities During Successful Trajectories</b>",
        showlegend=False,
    )
    fig.update_yaxes(title_text="Velocity (rad/s)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (rad/s)", row=1, col=2)
    fig.update_xaxes(title_text="Planning Methodology", row=1, col=1)
    fig.update_xaxes(title_text="Planning Methodology", row=1, col=2)

    output_path = os.path.join(output_dir, "comparative_articutool_velocities.html")
    fig.write_html(output_path)
    print(f"Saved Articutool velocity plot to {output_path}")


def plot_manifold_adherence(df: pd.DataFrame, output_dir: str):
    """
    Calculates and plots the 'Manifold Adherence' for each successful trajectory.
    Adherence is the percentage of waypoints within the preferred joint ranges
    for joints 2 and 3, as discovered during manifold exploration.
    """
    print("Generating manifold adherence plot...")

    success_df = df[df["status"] == "Success"].copy()

    if success_df.empty:
        print("Warning: No successful trials found. Skipping manifold adherence plot.")
        return

    adherence_scores = []
    for _, row in success_df.iterrows():
        waypoints = row.get("trajectory_metrics_waypoints_data")
        if not isinstance(waypoints, list) or not waypoints:
            continue

        adherent_waypoints = 0
        for wp in waypoints:
            pos = wp.get("jaco_positions_rad")
            if pos and len(pos) >= 3:
                # FIX: Correctly check the preferred ranges based on joint distribution analysis.
                # Joint 2 (pos[1]) preferred negative values.
                # Joint 3 (pos[2]) preferred positive values.
                if pos[1] < 0 and pos[2] > 0:
                    adherent_waypoints += 1

        adherence_scores.append(
            {
                "planning_mode": row["planning_mode"],
                "adherence_score": (adherent_waypoints / len(waypoints)) * 100,
            }
        )

    if not adherence_scores:
        print("Warning: Could not calculate any adherence scores. Skipping plot.")
        return

    adherence_df = pd.DataFrame(adherence_scores)

    fig = px.box(
        adherence_df,
        x="planning_mode",
        y="adherence_score",
        color="planning_mode",
        title="<b>Manifold Adherence of Successful Trajectories</b>",
        labels={
            "planning_mode": "Planning Methodology",
            "adherence_score": "Adherence Score (%)",
        },
        category_orders={
            "planning_mode": [
                "joint_unconstrained",
                "task_pos_unconstrained",
                "task_pos_constrained",
                "task_pos_yaw_constrained",
            ]
        },
    )
    fig.update_yaxes(range=[-5, 105])

    output_path = os.path.join(output_dir, "comparative_manifold_adherence.html")
    fig.write_html(output_path)
    print(f"Saved manifold adherence plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze results from the holistic benchmark."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="One or more JSON result files to analyze (e.g., 'output/*.json').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_analysis_plots",
        help="Directory to save the output HTML plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.json_files)

    plot_outcomes(df, args.output_dir)
    plot_planning_times(df, args.output_dir)
    plot_path_lengths(df, args.output_dir)
    plot_articutool_velocities(df, args.output_dir)
    plot_manifold_adherence(df, args.output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
