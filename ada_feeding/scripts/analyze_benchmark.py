import argparse
import pandas as pd
import plotly.express as px
import os


def load_and_prepare_data(articutool_file: str, baseline_file: str):
    """Loads and merges data from two .jsonl files for comparison."""
    try:
        df_articutool = pd.read_json(articutool_file, lines=True)
        df_articutool["mode"] = "Articutool"

        df_baseline = pd.read_json(baseline_file, lines=True)
        df_baseline["mode"] = "Baseline"
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure file paths are correct.")
        return None, None
    except ValueError as e:
        print(f"Error parsing JSONL file: {e}")
        return None, None

    df_trials = pd.concat([df_articutool, df_baseline], ignore_index=True)

    # Explode the stages data for granular analysis
    df_stages = (
        df_trials[["trial_id", "mode", "stages"]]
        .explode("stages")
        .reset_index(drop=True)
    )
    df_stages.dropna(subset=["stages"], inplace=True)
    stage_details = pd.json_normalize(df_stages["stages"])
    df_stages = pd.concat([df_stages.drop(columns=["stages"]), stage_details], axis=1)

    # Add a boolean success column for easier aggregation
    df_stages["is_success"] = (df_stages["status"] == "Success").astype(int)

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
        f"{'Mode':<12} | {'Total Trials':<15} | {'Successful':<12} | {'Success Rate (%)':<15}"
    )
    print("-" * 65)
    for _, row in summary.iterrows():
        print(
            f"{row['mode']:<12} | {row['count']:<15} | {int(row['sum']):<12} | {row['mean']:.1f}"
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
    for mode in ["Articutool", "Baseline"]:
        print(f"\n--- {mode} Results ---")
        mode_df = (
            final_summary[final_summary["mode"] == mode]
            .drop(columns="mode")
            .dropna(subset=["stage_name"])
        )
        print(mode_df.to_string(index=False, float_format="%.2f"))
    print("=" * 85)


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

    fig = px.bar(
        success_rates,
        x="stage_name",
        y="is_success",
        color="mode",
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

    fig = px.box(
        successful_stages,
        x="stage_name",
        y="planning_time_sec",
        color="mode",
        title="Planning Time for Successful Stages",
        labels={
            "stage_name": "Benchmark Stage",
            "planning_time_sec": "Planning Time (s)",
            "mode": "System",
        },
    )
    fig.write_html("planning_times.html")
    print("Saved planning time plot to planning_times.html")


def analyze_resting_stage_failures(df_stages: pd.DataFrame):
    """
    Analyzes and prints a summary of failure modes specifically for the 'Resting' stage.
    """
    print("\n" + "=" * 85)
    print("                      'Resting' Stage Failure Mode Analysis")
    print("=" * 85)

    # Filter for only the Articutool mode and the Resting stage
    df_resting = df_stages[
        (df_stages["mode"] == "Articutool") & (df_stages["stage_name"] == "Resting")
    ].copy()

    if df_resting.empty:
        print("No 'Resting' stage attempts found for the Articutool.")
        print("=" * 85)
        return

    # Filter for only the failures
    df_failures = df_resting[df_resting["is_success"] == 0]
    total_failures = len(df_failures)

    if total_failures == 0:
        print("No failures recorded for the 'Resting' stage. Great job!")
        print("=" * 85)
        return

    # Count the occurrences of each failure status
    failure_counts = df_failures["status"].value_counts()

    print(f"Total Failures in 'Resting' Stage for Articutool: {total_failures}\n")
    print("Breakdown of Failure Types:")
    for status, count in failure_counts.items():
        percentage = (count / total_failures) * 100
        print(f"- {status:<25}: {count:<5} ({percentage:.1f}%)")

    # Provide an interpretation based on the dominant failure mode
    if (
        "Planner Failure" in failure_counts
        and "Path Verification Failure" in failure_counts
    ):
        if (
            failure_counts["Planner Failure"]
            > failure_counts["Path Verification Failure"]
        ):
            print("\nInterpretation: Most failures are PLANNER_FAILURE.")
            print(
                "The planner is struggling to find any path, suggesting the goal or constraints are too strict."
            )
        else:
            print("\nInterpretation: Most failures are VERIFICATION_FAILURE.")
            print(
                "The planner finds paths, but they are kinematically infeasible for leveling."
            )
            print("This suggests the heuristic path constraint may be too loose.")
    elif "Planner Failure" in failure_counts:
        print("\nInterpretation: All failures are PLANNER_FAILURE.")
    elif "Path Verification Failure" in failure_counts:
        print("\nInterpretation: All failures are VERIFICATION_FAILURE.")

    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and compare benchmark results."
    )
    parser.add_argument(
        "articutool_file", type=str, help="Path to the Articutool benchmark JSONL file."
    )
    parser.add_argument(
        "baseline_file", type=str, help="Path to the Baseline benchmark JSONL file."
    )
    args = parser.parse_args()

    df_trials, df_stages = load_and_prepare_data(
        args.articutool_file, args.baseline_file
    )

    if df_trials is not None and df_stages is not None:
        generate_summary_table(df_trials)
        generate_stage_by_stage_summary(df_stages)
        plot_stage_success_rates(df_stages)
        plot_planning_times(df_stages)
        analyze_resting_stage_failures(df_stages)
