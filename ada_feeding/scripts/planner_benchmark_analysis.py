#!/usr/bin/env python3

"""
Analyzes the CSV output from the PlannerBenchmark script.
Generates summary statistics and comparative plots for different planners.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from typing import List, Dict, Any

# Define expected numeric columns that might need conversion and could contain NaN-like strings
NUMERIC_COLUMNS = [
    "elapsed_time_s",
    "jaco_plan_success",
    "path_length_total",
    "max_jaco_hand_roll_deviation_rad",
    "articutool_path_feasible",
    "articutool_num_infeasible_points",
    "articutool_min_pitch_rad",
    "articutool_max_pitch_rad",
    "articutool_avg_pitch_abs_rad",
    "articutool_pitch_range_used_percent",
    "articutool_min_roll_rad",
    "articutool_max_roll_rad",
    "articutool_avg_roll_abs_rad",
    "articutool_roll_range_used_percent",
]


def load_and_preprocess_data(csv_filepath: str) -> pd.DataFrame:
    """Loads data from CSV and preprocesses it."""
    print(f"Loading data from: {csv_filepath}")
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        raise
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

    print(f"Loaded {len(df)} rows.")

    # Identify path_length_joint columns dynamically
    path_length_joint_cols = [
        col for col in df.columns if col.startswith("path_length_j")
    ]
    all_numeric_cols = NUMERIC_COLUMNS + path_length_joint_cols

    for col in all_numeric_cols:
        if col in df.columns:
            # Replace common non-numeric placeholders with NaN before attempting conversion
            df[col] = df[col].replace(
                ["N/A", "nan", "inf", "-inf", "ERROR", "ERROR_NO_AT_METRICS", ""],
                np.nan,
            )
            # Attempt to convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(
                f"Warning: Expected numeric column '{col}' not found in CSV. Skipping."
            )

    # Ensure boolean-like columns are 0 or 1 (or NaN)
    for bool_col in ["jaco_plan_success", "articutool_path_feasible"]:
        if bool_col in df.columns:
            # After to_numeric, valid values should be 0.0, 1.0, or NaN.
            # We can keep them as float or convert to Int64 (which supports NaN)
            # For calculations like mean (success rate), float is fine.
            pass

    print("Data preprocessing complete.")
    return df


def calculate_summary_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculates summary statistics for each planner."""
    summary: Dict[str, Dict[str, Any]] = {}
    planners = df["planner_id"].unique()

    for planner in planners:
        planner_df = df[
            df["planner_id"] == planner
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        total_tasks = len(planner_df)
        if total_tasks == 0:
            summary[planner] = {"total_tasks": 0}
            continue

        # Jaco Plan Success
        jaco_successful_plans_df = planner_df[planner_df["jaco_plan_success"] == 1]
        num_jaco_successful = len(jaco_successful_plans_df)
        jaco_success_rate = (
            (num_jaco_successful / total_tasks) * 100 if total_tasks > 0 else 0
        )

        # Articutool Path Feasibility (of Jaco successful plans)
        articutool_feasible_df = pd.DataFrame()  # Ensure it's defined
        num_articutool_feasible = 0
        articutool_feasibility_rate = 0.0  # Default
        if num_jaco_successful > 0:
            articutool_feasible_df = jaco_successful_plans_df[
                jaco_successful_plans_df["articutool_path_feasible"] == 1
            ]
            num_articutool_feasible = len(articutool_feasible_df)
            articutool_feasibility_rate = (
                (num_articutool_feasible / num_jaco_successful) * 100
                if num_jaco_successful > 0
                else 0
            )

        # Overall Success (Jaco success AND Articutool feasible)
        overall_success_rate = (
            (num_articutool_feasible / total_tasks) * 100 if total_tasks > 0 else 0
        )

        # Planning Time
        avg_time_all = planner_df["elapsed_time_s"].mean()
        avg_time_jaco_succ = (
            jaco_successful_plans_df["elapsed_time_s"].mean()
            if num_jaco_successful > 0
            else np.nan
        )
        avg_time_overall_succ = (
            articutool_feasible_df["elapsed_time_s"].mean()
            if num_articutool_feasible > 0
            else np.nan
        )

        # Path Length
        avg_path_len_jaco_succ = (
            jaco_successful_plans_df["path_length_total"].mean()
            if num_jaco_successful > 0
            else np.nan
        )

        # Max Jaco Hand Roll Deviation
        avg_jaco_roll_dev_jaco_succ = (
            jaco_successful_plans_df["max_jaco_hand_roll_deviation_rad"].mean()
            if num_jaco_successful > 0
            else np.nan
        )

        # Articutool Metrics (for overall successful paths)
        avg_at_pitch_range_overall_succ = (
            articutool_feasible_df["articutool_pitch_range_used_percent"].mean()
            if num_articutool_feasible > 0
            else np.nan
        )
        avg_at_roll_range_overall_succ = (
            articutool_feasible_df["articutool_roll_range_used_percent"].mean()
            if num_articutool_feasible > 0
            else np.nan
        )
        avg_at_infeasible_pts_overall_succ = (
            articutool_feasible_df["articutool_num_infeasible_points"].mean()
            if num_articutool_feasible > 0
            else np.nan
        )

        summary[planner] = {
            "total_tasks": total_tasks,
            "num_jaco_successful": num_jaco_successful,
            "jaco_success_rate_percent": jaco_success_rate,
            "num_articutool_feasible_of_jaco_succ": num_articutool_feasible,  # Corrected key
            "articutool_feasibility_rate_percent": articutool_feasibility_rate,
            "num_overall_successful": num_articutool_feasible,  # Same as above, but for clarity in overall rate
            "overall_success_rate_percent": overall_success_rate,
            "avg_time_all_s": avg_time_all,
            "avg_time_jaco_succ_s": avg_time_jaco_succ,
            "avg_time_overall_succ_s": avg_time_overall_succ,
            "avg_path_len_jaco_succ": avg_path_len_jaco_succ,
            "avg_jaco_roll_dev_jaco_succ_rad": avg_jaco_roll_dev_jaco_succ,
            "avg_at_pitch_range_overall_succ_percent": avg_at_pitch_range_overall_succ,
            "avg_at_roll_range_overall_succ_percent": avg_at_roll_range_overall_succ,
            "avg_at_infeasible_pts_overall_succ": avg_at_infeasible_pts_overall_succ,
        }
    return summary


def print_summary_report(summary_stats: Dict[str, Dict[str, Any]]):
    """Prints a formatted summary report to the console."""
    print("\n--- Benchmark Analysis Report ---")
    for planner, stats in summary_stats.items():
        print(f"\nPlanner: {planner} (Total Tasks: {stats.get('total_tasks', 0)})")
        if stats.get("total_tasks", 0) == 0:
            print("  No data for this planner.")
            continue

        print(
            f"  Jaco Plan Success Rate: {stats.get('jaco_success_rate_percent', 0):.2f}% ({stats.get('num_jaco_successful', 0)}/{stats.get('total_tasks', 0)})"
        )
        if stats.get("num_jaco_successful", 0) > 0:
            print(
                f"  Articutool Feasibility (of Jaco succ.): {stats.get('articutool_feasibility_rate_percent', 0):.2f}% ({stats.get('num_articutool_feasible_of_jaco_succ', 0)}/{stats.get('num_jaco_successful', 0)})"
            )
        else:
            print(
                "  Articutool Feasibility (of Jaco succ.): N/A (No Jaco successful plans)"
            )
        print(
            f"  Overall Task Success Rate: {stats.get('overall_success_rate_percent', 0):.2f}% ({stats.get('num_overall_successful', 0)}/{stats.get('total_tasks', 0)})"
        )

        print(
            f"  Avg. Planning Time (all attempts): {stats.get('avg_time_all_s', np.nan):.3f} s"
        )
        print(
            f"  Avg. Planning Time (Jaco successful): {stats.get('avg_time_jaco_succ_s', np.nan):.3f} s"
        )
        print(
            f"  Avg. Planning Time (Overall successful): {stats.get('avg_time_overall_succ_s', np.nan):.3f} s"
        )

        print(
            f"  Avg. Path Length (Jaco successful): {stats.get('avg_path_len_jaco_succ', np.nan):.3f}"
        )
        print(
            f"  Avg. Max Jaco Hand Roll Dev (Jaco successful): {stats.get('avg_jaco_roll_dev_jaco_succ_rad', np.nan):.3f} rad"
        )

        if stats.get("num_overall_successful", 0) > 0:
            print(
                f"  Avg. Articutool Pitch Range Used (Overall successful): {stats.get('avg_at_pitch_range_overall_succ_percent', np.nan):.2f}%"
            )
            print(
                f"  Avg. Articutool Roll Range Used (Overall successful): {stats.get('avg_at_roll_range_overall_succ_percent', np.nan):.2f}%"
            )
            print(
                f"  Avg. Articutool Infeasible Points (Overall successful): {stats.get('avg_at_infeasible_pts_overall_succ', np.nan):.2f}"
            )
        else:
            print("  Articutool Performance Metrics (Overall successful): N/A")
    print("-----------------------------")


def generate_plots(df: pd.DataFrame, output_dir: str):
    """Generates and saves comparative plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory for plots: {output_dir}")

    sns.set_theme(style="whitegrid")
    planners = df["planner_id"].unique()

    # --- Success Rate Plots ---
    success_data = []
    for planner in planners:
        planner_df = df[df["planner_id"] == planner]
        total = len(planner_df)
        if total == 0:
            continue
        jaco_succ = planner_df["jaco_plan_success"].sum()

        # For Articutool feasibility, only consider Jaco successful plans
        jaco_succ_df = planner_df[planner_df["jaco_plan_success"] == 1]
        at_feasible_of_jaco_succ = (
            jaco_succ_df["articutool_path_feasible"].sum()
            if not jaco_succ_df.empty
            else 0
        )

        success_data.append(
            {
                "planner": planner,
                "type": "Jaco Plan Success",
                "rate": (jaco_succ / total) * 100 if total else 0,
            }
        )
        success_data.append(
            {
                "planner": planner,
                "type": "Articutool Feasible (of Jaco Succ.)",
                "rate": (
                    (at_feasible_of_jaco_succ / jaco_succ) * 100 if jaco_succ else 0
                ),
            }
        )
        success_data.append(
            {
                "planner": planner,
                "type": "Overall Success",
                "rate": (at_feasible_of_jaco_succ / total) * 100 if total else 0,
            }
        )

    success_df = pd.DataFrame(success_data)

    plt.figure(figsize=(12, 7))
    sns.barplot(x="planner", y="rate", hue="type", data=success_df, palette="viridis")
    plt.title("Planner Success Rates")
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Planner ID")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Success Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rates.png"))
    plt.close()
    print(f"Saved plot: success_rates.png to {output_dir}")

    # --- Distribution Plots (Box Plots) ---
    # Filter for Jaco successful plans for these metrics
    jaco_successful_df = df[df["jaco_plan_success"] == 1].copy()
    # Filter for overall successful plans for Articutool specific metrics
    overall_successful_df = jaco_successful_df[
        jaco_successful_df["articutool_path_feasible"] == 1
    ].copy()

    plot_metrics = [
        ("elapsed_time_s", "Planning Time (Jaco Successful) (s)", jaco_successful_df),
        ("path_length_total", "Path Length (Jaco Successful)", jaco_successful_df),
        (
            "max_jaco_hand_roll_deviation_rad",
            "Max Jaco Hand Roll Dev (Jaco Successful) (rad)",
            jaco_successful_df,
        ),
        (
            "articutool_pitch_range_used_percent",
            "Articutool Pitch Range Used (Overall Successful) (%)",
            overall_successful_df,
        ),
        (
            "articutool_roll_range_used_percent",
            "Articutool Roll Range Used (Overall Successful) (%)",
            overall_successful_df,
        ),
    ]

    for metric_col, title, data_subset in plot_metrics:
        if (
            metric_col not in data_subset.columns
            or data_subset[metric_col].isnull().all()
        ):
            print(f"Skipping plot for '{title}': column missing or all NaN.")
            continue
        if data_subset.empty:
            print(f"Skipping plot for '{title}': no data after filtering.")
            continue

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="planner_id", y=metric_col, data=data_subset, palette="pastel")
        plt.title(title)
        plt.ylabel(metric_col.replace("_", " ").title())
        plt.xlabel("Planner ID")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_col}_distribution.png"))
        plt.close()
        print(f"Saved plot: {metric_col}_distribution.png to {output_dir}")

    print("Plot generation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Planner Benchmark CSV results."
    )
    parser.add_argument(
        "csv_filepath", type=str, help="Path to the benchmark CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_analysis_plots",
        help="Directory to save generated plots (default: benchmark_analysis_plots).",
    )
    args = parser.parse_args()

    try:
        data_df = load_and_preprocess_data(args.csv_filepath)
        summary_stats = calculate_summary_stats(data_df)
        print_summary_report(summary_stats)
        generate_plots(data_df, args.output_dir)
        print("\nAnalysis complete.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
