import argparse
import json
import pandas as pd
from dataclasses import asdict


def analyze_benchmark_results(file_path: str):
    """
    Loads benchmark data, calculates key metrics, and prints a summary report,
    including statistics grouped by the stage of failure.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing file: {e}")
        return

    if not data:
        print("No data found in the benchmark file.")
        return

    # --- 1. Flatten the data for easier analysis with pandas ---
    # Each row in this DataFrame will represent a single stage from a single trial.
    flat_stages = []
    trial_characteristics = []

    for trial in data:
        trial_characteristics.append(
            {
                "trial_id": trial.get("trial_id"),
                "end_to_end_success": trial.get("end_to_end_success"),
                **trial.get("scene_characteristics", {}),
            }
        )
        for stage in trial.get("stages", []):
            flat_record = {
                "trial_id": trial.get("trial_id"),
                "end_to_end_success": trial.get("end_to_end_success"),
                "stage_name": stage.get("stage_name"),
                "status": stage.get("status"),
                "planning_time_sec": stage.get("planning_time_sec"),
                "path_length_m": stage.get("trajectory_path_length_m"),
            }
            # Add custom metrics if they exist
            if stage.get("custom_metrics"):
                flat_record.update(stage["custom_metrics"])
            flat_stages.append(flat_record)

    df_stages = pd.DataFrame(flat_stages)
    df_trials = pd.DataFrame(trial_characteristics)

    # --- 2. Print High-Level Summary ---
    num_trials = len(data)
    successful_trials = df_trials["end_to_end_success"].sum()
    overall_success_rate = (
        (successful_trials / num_trials) * 100 if num_trials > 0 else 0
    )

    print("\n" + "=" * 60)
    print("           BENCHMARK ANALYSIS REPORT")
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Total Trials: {num_trials}")
    print(
        f"End-to-End Success Rate: {overall_success_rate:.1f}% ({successful_trials}/{num_trials})"
    )
    print("-" * 60)

    if not df_stages.empty:
        failed_stages_df = df_stages[df_stages["status"] != "Success"]
        stage_failure_counts = failed_stages_df["stage_name"].value_counts()
        print("\nFailure Count by Stage:")
        if not stage_failure_counts.empty:
            print(stage_failure_counts.to_string())
        else:
            print("No stage failures recorded.")
        print("-" * 60)

        # --- Failure analysis by TYPE for the most problematic stage ---
        if not stage_failure_counts.empty:
            most_problematic_stage = stage_failure_counts.index[0]
            print(
                f"\nFailure Breakdown for Most Problematic Stage ('{most_problematic_stage}'):"
            )
            problem_stage_df = failed_stages_df[
                failed_stages_df["stage_name"] == most_problematic_stage
            ]
            status_counts = problem_stage_df["status"].value_counts()
            print(status_counts.to_string())
        print("=" * 60)

    # --- 3. Generate Comprehensive Statistical Summaries (Binary Pass/Fail) ---
    if not df_trials.empty:
        # Define the columns we want to summarize
        characteristic_cols = [
            col
            for col in df_trials.columns
            if col not in ["trial_id", "end_to_end_success"]
        ]

        # Separate successful and failed trials
        successful_df = df_trials[df_trials["end_to_end_success"] == True]

        print("\n           COMPREHENSIVE STATISTICAL SUMMARY")
        print("-" * 60)

        if not successful_df.empty:
            print(
                f"\n--- For SUCCESSFUL End-to-End Trials ({len(successful_df)} trials) ---"
            )
            # Use .describe() to get summary statistics, transpose for readability
            print(successful_df[characteristic_cols].describe().transpose().to_string())
        else:
            print("\n--- No SUCCESSFUL End-to-End Trials to Summarize ---")

    # --- 4. Generate Statistical Summaries by Failure Stage ---
    if not df_stages.empty and not df_trials.empty:
        # Find the first failing stage for each failed trial
        failed_stages_df = df_stages[df_stages["status"] != "Success"]
        first_failures = failed_stages_df.loc[
            failed_stages_df.groupby("trial_id").head(1).index
        ]
        first_failures = first_failures.rename(columns={"stage_name": "failure_stage"})

        # Merge this information back with the trial characteristics
        trials_with_failure_stage = pd.merge(
            df_trials,
            first_failures[["trial_id", "failure_stage"]],
            on="trial_id",
            how="left",
        )

        failure_stages = trials_with_failure_stage["failure_stage"].dropna().unique()

        print("\n\n" + "=" * 60)
        print("     STATISTICAL SUMMARY BY STAGE OF FAILURE")
        print("=" * 60)

        for stage in failure_stages:
            stage_specific_failures_df = trials_with_failure_stage[
                trials_with_failure_stage["failure_stage"] == stage
            ]
            if not stage_specific_failures_df.empty:
                print(
                    f"\n--- For Trials Failing at Stage: '{stage}' ({len(stage_specific_failures_df)} trials) ---"
                )
                print(
                    stage_specific_failures_df[characteristic_cols]
                    .describe()
                    .transpose()
                    .to_string()
                )
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze end-to-end benchmark results."
    )
    parser.add_argument(
        "benchmark_file", type=str, help="Path to the benchmark JSON output file."
    )
    args = parser.parse_args()

    analyze_benchmark_results(args.benchmark_file)
