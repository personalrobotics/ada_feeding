import argparse
import json
import pandas as pd


def analyze_benchmark_results(file_path: str):
    """
    Loads benchmark data, calculates key metrics, and prints a summary report.
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
    for trial in data:
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

    df = pd.DataFrame(flat_stages)

    # --- 2. Calculate and Print Key Metrics ---
    num_trials = len(data)
    successful_trials = sum(1 for trial in data if trial.get("end_to_end_success"))
    overall_success_rate = (
        (successful_trials / num_trials) * 100 if num_trials > 0 else 0
    )

    print("\n" + "=" * 50)
    print("      BENCHMARK ANALYSIS REPORT")
    print("=" * 50)
    print(f"File: {file_path}")
    print(f"Total Trials: {num_trials}")
    print(
        f"End-to-End Success Rate: {overall_success_rate:.1f}% ({successful_trials}/{num_trials})"
    )
    print("-" * 50)

    if not df.empty:
        # --- Failure analysis by STAGE ---
        failed_stages_df = df[df["status"] != "Success"]
        stage_failure_counts = failed_stages_df["stage_name"].value_counts()
        print("\nFailure Count by Stage:")
        if not stage_failure_counts.empty:
            print(stage_failure_counts.to_string())
        else:
            print("No stage failures recorded.")
        print("-" * 50)

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
            print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze end-to-end benchmark results."
    )
    parser.add_argument(
        "benchmark_file", type=str, help="Path to the benchmark JSON output file."
    )
    args = parser.parse_args()

    analyze_benchmark_results(args.benchmark_file)
