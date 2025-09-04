import argparse
import pandas as pd
import plotly.express as px
import os
import math


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
