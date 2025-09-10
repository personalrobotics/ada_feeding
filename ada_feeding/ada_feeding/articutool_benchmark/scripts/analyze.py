import argparse
import pandas as pd
import plotly.express as px
import os
import math
import numpy as np

import matplotlib.pyplot as plt

ARTICUTOOL_MAX_VELOCITY_RAD_S = 4.0


def load_and_prepare_data(articutool_file: str, baseline_files: list):
    """Loads and merges data from multiple .jsonl files for comparison."""
    try:
        df_articutool = pd.read_json(articutool_file, lines=True)
        df_articutool["mode"] = "Articutool"

        all_dfs = [df_articutool]
        if baseline_files:
            for baseline_file in baseline_files:
                df_baseline = pd.read_json(baseline_file, lines=True)
                if "6dof_baseline" in baseline_file:
                    df_baseline["mode"] = "6dof_baseline"
                elif "8dof_baseline" in baseline_file:
                    df_baseline["mode"] = "8dof_baseline"
                else:
                    # Fallback for old naming convention
                    df_baseline["mode"] = "Baseline"
                all_dfs.append(df_baseline)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure file paths are correct.")
        return None, None
    except ValueError as e:
        print(f"Error parsing JSONL file: {e}")
        return None, None

    # Process each dataframe separately to handle potentially different nested structures
    def process_df(df):
        # Explode the stages data for granular analysis
        df_exploded = (
            df[["trial_id", "mode", "stages"]].explode("stages").reset_index(drop=True)
        )
        df_exploded.dropna(subset=["stages"], inplace=True)
        stage_details = pd.json_normalize(df_exploded["stages"])
        df_processed = pd.concat(
            [df_exploded.drop(columns=["stages"]), stage_details], axis=1
        )
        # Add a boolean success column for easier aggregation
        df_processed["is_success"] = (df_processed["status"] == "Success").astype(int)
        return df_processed

    list_of_stage_dfs = [process_df(df) for df in all_dfs]
    df_stages = pd.concat(list_of_stage_dfs, ignore_index=True)
    df_trials = pd.concat(all_dfs, ignore_index=True)

    return df_trials, df_stages


def generate_summary_table(df_trials: pd.DataFrame):
    """Prints a high-level comparative summary table."""
    summary = (
        df_trials.groupby("mode")["end_to_end_success"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    summary["mean"] = summary["mean"] * 100

    print("\n" + "=" * 65)
    print("                 HIGH-LEVEL BENCHMARK SUMMARY")
    print("=" * 65)
    print(
        f"{'Mode':<25} | {'Total Trials':<15} | {'Successful':<12} | {'Success Rate (%)'}"
    )
    print("-" * 65)
    for _, row in summary.iterrows():
        print(
            f"{row['mode']:<25} | {row['count']:<15} | {int(row['sum']):<12} | {row['mean']:.1f}"
        )
    print("=" * 65)


def generate_stage_by_stage_summary(df_stages: pd.DataFrame):
    """Calculates and prints a detailed, stage-by-stage aggregate summary."""
    # Calculate success rate on all attempts
    success_rates = (
        df_stages.groupby(["mode", "stage_name"])["is_success"]
        .agg(["mean", "count"])
        .reset_index()
    )
    success_rates["mean"] *= 100
    success_rates = success_rates.rename(
        columns={"mean": "Success Rate (%)", "count": "Attempts"}
    )

    # Calculate other metrics only on successful attempts
    df_success = df_stages[df_stages["is_success"] == 1].copy()
    aggregations = {
        "planning_time_sec": "mean",
        "trajectory_path_length_m": "mean",
        "total_joint_travel_rad": "mean",
    }
    agg_results = (
        df_success.groupby(["mode", "stage_name"]).agg(aggregations).reset_index()
    )
    agg_results = agg_results.rename(
        columns={
            "planning_time_sec": "Avg Plan Time (s)",
            "trajectory_path_length_m": "Avg Path Length (m)",
            "total_joint_travel_rad": "Avg Joint Travel (rad)",
        }
    )

    # Merge the two dataframes
    final_summary = pd.merge(
        success_rates, agg_results, on=["mode", "stage_name"], how="left"
    )

    # Define a consistent order for stages
    stage_order = [
        "HomeToAbovePlate",
        "PreAcquisition",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelTool",
        "Resting",
        "Staging",
        "Presentation",
    ]
    final_summary["stage_name"] = pd.Categorical(
        final_summary["stage_name"], categories=stage_order, ordered=True
    )
    final_summary = final_summary.sort_values(["mode", "stage_name"])

    print("\n" + "=" * 85)
    print("                           STAGE-BY-STAGE AGGREGATE RESULTS")
    print("=" * 85)
    for mode in final_summary["mode"].unique():
        print(f"\n--- {mode} Results ---")
        mode_df = (
            final_summary[final_summary["mode"] == mode]
            .copy()
            .drop(columns="mode")
            .dropna(subset=["stage_name"])
        )
        print(mode_df.to_string(index=False, float_format="%.2f"))
    print("=" * 85)


def analyze_stage_failures(df_stages: pd.DataFrame):
    """
    Analyzes and prints a summary of failure modes for each stage,
    focusing on the Articutool mode.
    """
    print("\n" + "=" * 85)
    print("                      DETAILED STAGE FAILURE ANALYSIS (Articutool Mode)")
    print("=" * 85)

    # Define a consistent order for stages to analyze
    stage_order = [
        "HomeToAbovePlate",
        "PreAcquisition",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelTool",
        "Resting",
        "Staging",
        "Presentation",
    ]

    # Filter to only stages that are actually present in the dataframe
    present_stages = [s for s in stage_order if s in df_stages["stage_name"].unique()]

    for stage_name in present_stages:
        print(f"\n--- Analysis for Stage: '{stage_name}' ---")

        # Filter for the specific stage and mode
        df_stage_mode = df_stages[
            (df_stages["mode"] == "Articutool")
            & (df_stages["stage_name"] == stage_name)
        ].copy()

        if df_stage_mode.empty:
            print(f"No attempts found for this stage.")
            continue

        # Filter for only the failures in this stage
        df_failures = df_stage_mode[df_stage_mode["is_success"] == 0]
        total_failures = len(df_failures)

        if total_failures == 0:
            print("No failures recorded for this stage. Great job!")
            continue

        # Count the occurrences of each failure status
        failure_counts = df_failures["status"].value_counts()

        print(f"Total Failures: {total_failures}\n")
        print("Breakdown of Failure Types:")
        for status, count in failure_counts.items():
            percentage = (count / total_failures) * 100
            print(f"- {status:<25}: {count:<5} ({percentage:.1f}%)")

        # Provide an interpretation based on the dominant failure modes
        if (
            "Planner Failure" in failure_counts
            or "Verification Failure" in failure_counts
        ):
            # This interpretation is particularly relevant for transport stages like Resting/Staging
            planner_fails = failure_counts.get("Planner Failure", 0)
            verification_fails = failure_counts.get("Verification Failure", 0)

            print("\nInterpretation:")
            if planner_fails > verification_fails:
                print("-> Dominant failure is PLANNER_FAILURE.")
                print(
                    "   The planner is struggling to find any path, suggesting the goal or constraints are too strict."
                )
            elif verification_fails > planner_fails:
                print("-> Dominant failure is VERIFICATION_FAILURE.")
                print(
                    "   The planner finds paths, but they are kinematically infeasible for leveling."
                )
                print(
                    "   This suggests the heuristic path constraint may be too loose."
                )
            else:
                print(
                    "-> Planner and Verification failures are balanced or other failure types dominate."
                )

    print("\n" + "=" * 85)


def plot_stage_success_rates(df_stages: pd.DataFrame):
    """Generates a grouped bar chart comparing success rates for each stage."""
    stage_order = [
        "HomeToAbovePlate",
        "PreAcquisition",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelTool",
        "Resting",
        "Staging",
        "Presentation",
    ]
    success_rates = (
        df_stages.groupby(["mode", "stage_name"])["is_success"].mean().reset_index()
    )
    success_rates["is_success"] *= 100
    success_rates["stage_name"] = pd.Categorical(
        success_rates["stage_name"], categories=stage_order, ordered=True
    )
    success_rates = success_rates.dropna(subset=["stage_name"]).sort_values(
        "stage_name"
    )

    mode_order = ["Articutool", "6dof_baseline", "8dof_baseline"]
    present_modes = [
        mode for mode in mode_order if mode in success_rates["mode"].unique()
    ]

    fig = px.bar(
        success_rates,
        x="stage_name",
        y="is_success",
        color="mode",
        category_orders={"mode": present_modes},
        barmode="group",
        text_auto=".1f",
        title="Success Rate by Stage",
        labels={
            "stage_name": "Benchmark Stage",
            "is_success": "Success Rate (%)",
            "mode": "System",
        },
    )
    fig.update_traces(textangle=0, textposition="outside")
    fig.update_yaxes(range=[0, 105])
    fig.write_html("stage_success_rates.html")
    print("\nSaved stage success rate plot to stage_success_rates.html")


def plot_planning_times(df_stages: pd.DataFrame):
    """Generates a box plot comparing planning times for successful stages."""
    successful_stages = df_stages[df_stages["is_success"] == 1].copy()
    stage_order = [
        "HomeToAbovePlate",
        "PreAcquisition",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelTool",
        "Resting",
        "Staging",
        "Presentation",
    ]
    successful_stages["stage_name"] = pd.Categorical(
        successful_stages["stage_name"], categories=stage_order, ordered=True
    )
    successful_stages = successful_stages.dropna(subset=["stage_name"]).sort_values(
        "stage_name"
    )

    mode_order = ["Articutool", "6dof_baseline", "8dof_baseline"]
    present_modes = [
        mode for mode in mode_order if mode in successful_stages["mode"].unique()
    ]

    fig = px.box(
        successful_stages,
        x="stage_name",
        y="planning_time_sec",
        color="mode",
        category_orders={"mode": present_modes},
        title="Planning Time for Successful Stages",
        labels={
            "stage_name": "Benchmark Stage",
            "planning_time_sec": "Planning Time (s)",
            "mode": "System",
        },
    )
    fig.write_html("planning_times.html")
    print("Saved planning time plot to planning_times.html")


def plot_required_velocities(df_stages: pd.DataFrame):
    """
    Generates a box plot of the max required Articutool velocities for
    transport stages, relative to the motor limit.
    """
    # Filter for the relevant stages that have the dynamic check data.
    transport_stages = df_stages[
        df_stages["stage_name"].isin(["Resting", "Staging"])
    ].copy()

    velocity_col_name = "custom_metrics.max_required_velocity_rad_s"

    # Check if the dynamic verification data exists in the DataFrame
    if velocity_col_name not in transport_stages.columns:
        print(
            "\nNo dynamic verification data ('max_required_velocity_rad_s') found to plot."
        )
        return

    # Rename the column for easier access in Plotly
    transport_stages.rename(
        columns={velocity_col_name: "max_required_velocity"}, inplace=True
    )

    # Drop rows where the metric is null (e.g., planner failures before verification)
    transport_stages.dropna(subset=["max_required_velocity"], inplace=True)

    if transport_stages.empty:
        print("\nNo dynamic verification data found to plot required velocities.")
        return

    fig = px.box(
        transport_stages,
        x="stage_name",
        y="max_required_velocity",
        points="all",  # Show all individual data points
        title="Max Required Articutool Velocity During Transport",
        labels={
            "stage_name": "Benchmark Stage",
            "max_required_velocity": "Max Required Velocity (rad/s)",
        },
    )

    # Add a horizontal line indicating the maximum velocity threshold.
    fig.add_hline(
        y=ARTICUTOOL_MAX_VELOCITY_RAD_S,
        line_dash="dash",
        line_color="red",
        annotation_text="Motor Velocity Limit",
        annotation_position="bottom right",
    )

    # Adjust the y-axis to give some space above the limit line for clarity.
    fig.update_yaxes(range=[0, ARTICUTOOL_MAX_VELOCITY_RAD_S + 1.0])

    fig.write_html("required_velocities.html")
    print("\nSaved required velocity plot to required_velocities.html")


def analyze_6dof_reachability(df_trials: pd.DataFrame, df_stages: pd.DataFrame):
    """
    Analyzes and plots 6-DOF reachability failures against horizontal distance
    and the polar approach angle (phi).
    """
    preacq_6dof = df_stages[
        (df_stages["mode"] == "6dof_baseline")
        & (df_stages["stage_name"] == "PreAcquisition")
    ]

    if preacq_6dof.empty:
        print("\nNo 6-DOF PreAcquisition data found to analyze.")
        return

    success_map = preacq_6dof.set_index("trial_id")["is_success"]
    analysis_df = df_trials[df_trials["mode"] == "6dof_baseline"].copy()
    analysis_df["Success"] = (
        analysis_df["trial_id"].map(success_map).map({1: "Success", 0: "Failure"})
    )
    analysis_df.dropna(subset=["Success"], inplace=True)

    try:
        analysis_df["in_food_dist_2d"] = analysis_df["scene_characteristics"].apply(
            lambda x: x.get("in_food_pose_dist_2d")
        )
        analysis_df["in_food_polar_rad"] = analysis_df["scene_characteristics"].apply(
            lambda x: x.get("in_food_sampled_polar_angle_rad")
        )

        # Convert polar angle to degrees for more intuitive plotting
        analysis_df["in_food_polar_deg"] = np.rad2deg(analysis_df["in_food_polar_rad"])

        analysis_df.dropna(
            subset=["in_food_dist_2d", "in_food_polar_deg"], inplace=True
        )

    except (KeyError, TypeError):
        print("\nCould not extract required characteristics for reachability plot.")
        return

    if analysis_df.empty:
        print("\nNo valid data left to plot after extraction.")
        return

    fig = px.scatter(
        analysis_df,
        x="in_food_dist_2d",
        y="in_food_polar_deg",
        color="Success",
        color_discrete_map={"Success": "green", "Failure": "red"},
        title="6-DOF Baseline: Reachability vs. Distance and Approach Angle",
        labels={
            "in_food_dist_2d": "Horizontal Distance of InFood Pose (m)",
            "in_food_polar_deg": "Polar Approach Angle (Phi, degrees)",
        },
        hover_data=["trial_id"],
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.write_html("6dof_reachability_analysis.html")
    print("\nSaved 6-DOF reachability analysis plot to 6dof_reachability_analysis.html")


def plot_end_to_end_success(df_trials: pd.DataFrame):
    """Generates the primary 'hero' bar chart using matplotlib."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    summary = df_trials.groupby("mode")["end_to_end_success"].agg("mean").reset_index()
    summary["end_to_end_success"] *= 100
    mode_order = ["6dof_baseline", "8dof_baseline", "Articutool"]
    summary["mode"] = pd.Categorical(
        summary["mode"], categories=mode_order, ordered=True
    )
    summary = summary.sort_values("mode")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "Articutool": "#19878C",
        "6dof_baseline": "#C8C8C8",
        "8dof_baseline": "#969696",
    }
    bars = ax.bar(
        summary["mode"],
        summary["end_to_end_success"],
        color=[colors[m] for m in summary["mode"]],
    )
    ax.set_title(
        "Articutool's Decoupled Approach Improves End-to-End Success",
        fontsize=16,
        weight="bold",
    )
    ax.set_ylabel("End-to-End Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig("end_to_end_success.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("\nSaved end-to-end success plot (matplotlib) to end_to_end_success.pdf")


def plot_stage_survival(df_stages: pd.DataFrame):
    """Generates a 'survival plot' using matplotlib."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    stage_order = [
        "HomeToAbovePlate",
        "PreAcquisition",
        "AbovePlateToAboveFood",
        "AboveFoodToInFood",
        "LevelTool",
        "Resting",
        "Staging",
        "Presentation",
    ]
    success_rates = df_stages.groupby(["mode", "stage_name"])["is_success"].mean()
    survival_df = success_rates.groupby(level="mode").cumprod().reset_index()
    survival_df.rename(columns={"is_success": "survival_rate"}, inplace=True)
    survival_df["survival_rate"] *= 100
    start_points = pd.DataFrame(
        {
            "mode": survival_df["mode"].unique(),
            "stage_name": "Start",
            "survival_rate": 100.0,
        }
    )
    full_survival_df = pd.concat([start_points, survival_df], ignore_index=True)
    full_survival_df["stage_name"] = pd.Categorical(
        full_survival_df["stage_name"], categories=["Start"] + stage_order, ordered=True
    )
    full_survival_df = full_survival_df.sort_values("stage_name")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "Articutool": "#19878C",
        "6dof_baseline": "#C8C8C8",
        "8dof_baseline": "#969696",
    }
    for mode, group in full_survival_df.groupby("mode"):
        ax.plot(
            group["stage_name"],
            group["survival_rate"],
            marker="o",
            linestyle="-",
            label=mode,
            color=colors.get(mode),
        )
    ax.set_title(
        "Baselines Fail at Critical Dexterity and Planning Complexity Bottlenecks",
        fontsize=16,
        weight="bold",
    )
    ax.set_ylabel("Trials Remaining Successful (%)")
    ax.set_xlabel("Benchmark Stage")
    ax.set_ylim(-5, 105)
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="System")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.annotate(
        "6-DOF Dexterity Failure",
        xy=("LevelTool", 27),
        xytext=("AboveFoodToInFood", 45),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
        fontsize=12,
        ha="center",
    )
    ax.annotate(
        "8-DOF Planning Complexity Failure",
        xy=("Resting", 21),
        xytext=("LevelTool", 40),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
        fontsize=12,
        ha="center",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig("stage_survival_plot.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Saved stage survival plot (matplotlib) to stage_survival_plot.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and compare benchmark results."
    )
    parser.add_argument(
        "--articutool_file",
        type=str,
        required=True,
        help="Path to the Articutool benchmark JSONL file.",
    )
    parser.add_argument(
        "--baseline_files",
        type=str,
        nargs="+",
        help="Paths to one or more baseline JSONL files (e.g., 6dof, 8dof).",
    )
    args = parser.parse_args()

    df_trials, df_stages = load_and_prepare_data(
        args.articutool_file, args.baseline_files
    )

    if df_trials is not None and df_stages is not None:
        generate_summary_table(df_trials)
        generate_stage_by_stage_summary(df_stages)
        analyze_stage_failures(df_stages)
        plot_stage_success_rates(df_stages)
        plot_planning_times(df_stages)
        plot_required_velocities(df_stages)
        analyze_6dof_reachability(df_trials, df_stages)
        plot_end_to_end_success(df_trials)
        plot_stage_survival(df_stages)
