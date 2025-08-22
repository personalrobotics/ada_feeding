import argparse
import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_visualizations(df_trials: pd.DataFrame, file_path: str):
    """
    Generates interactive Plotly visualizations with a dropdown for stage-specific analysis.
    """
    if df_trials.empty:
        print("Skipping visualization, no trial data to plot.")
        return

    # --- 1. Load and Prepare Granular Stage Data ---
    try:
        with open(file_path, "r") as f:
            full_data = [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading or parsing file for visualization: {e}")
        return

    # Create a DataFrame with one row per stage to get granular status
    stages_records = []
    for trial in full_data:
        for stage in trial.get("stages", []):
            stages_records.append(
                {
                    "trial_id": trial["trial_id"],
                    "stage_name": stage["stage_name"],
                    "status": (
                        1 if stage["status"] == "Success" else 0
                    ),  # Use 1 for success, 0 for failure
                }
            )
    df_stages = pd.DataFrame(stages_records)

    # Pivot the data so each trial has a column for each stage's status
    df_stage_status = df_stages.pivot(
        index="trial_id", columns="stage_name", values="status"
    )
    df_stage_status.columns = [
        f"status_{col}" for col in df_stage_status.columns
    ]  # Rename columns

    # --- 2. Prepare Data for 3D Workspace Plot ---
    plot_data = []
    for trial in full_data:
        trial_id = trial.get("trial_id")
        # Merge trial characteristics and stage statuses
        characteristics = df_trials[df_trials["trial_id"] == trial_id]
        if characteristics.empty:
            continue

        trial_info = characteristics.iloc[0].to_dict()
        if trial_id in df_stage_status.index:
            trial_info.update(df_stage_status.loc[trial_id].to_dict())

        for pose_name, pose_data in trial.get("scene_poses", {}).items():
            if pose_name in ["food_pose", "mouth_pose", "resting_pose"]:
                pos = pose_data.get("position")
                if not isinstance(pos, list) or len(pos) != 3:
                    continue

                # Add all info to the record for plotting
                record = {"x": pos[0], "y": pos[1], "z": pos[2], "pose_name": pose_name}
                record.update(trial_info)
                plot_data.append(record)

    if not plot_data:
        print("No valid pose data found to generate visualizations.")
        return
    vis_df = pd.DataFrame(plot_data)

    # --- 3. Create Figure and Traces ---
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scatter3d"}]],
        subplot_titles=("3D Workspace Visualization",),
    )

    # Define all possible status columns we can filter by
    status_cols = {"End-to-End": "end_to_end_success"}
    for col in df_stage_status.columns:
        stage_name = col.replace("status_", "")
        status_cols[f"Stage: {stage_name}"] = col

    symbols = {"food_pose": "circle", "mouth_pose": "square", "resting_pose": "diamond"}

    # Create all traces upfront, most will be hidden initially
    for status_name, status_col in status_cols.items():
        for pose_name, symbol in symbols.items():
            for success_val, color in zip([1, 0], ["green", "red"]):
                is_success = success_val == 1
                df_subset = vis_df[
                    (vis_df["pose_name"] == pose_name)
                    & (vis_df[status_col] == success_val)
                ]
                if not df_subset.empty:
                    fig.add_trace(
                        go.Scatter3d(
                            x=df_subset["x"],
                            y=df_subset["y"],
                            z=df_subset["z"],
                            mode="markers",
                            marker=dict(
                                color=color, symbol=symbol, size=5, opacity=0.7
                            ),
                            # Make only the default view (End-to-End) visible initially
                            visible=(status_name == "End-to-End"),
                            name=f"{'Success' if is_success else 'Failure'} ({pose_name.replace('_pose', '')})",
                            customdata=df_subset.to_dict("records"),
                            hovertemplate="<b>Trial ID: %{customdata.trial_id}</b><br>Pose: %{customdata.pose_name}<br>Success: "
                            + ("Yes" if is_success else "No")
                            + "<br>Distance: %{customdata.food_mouth_distance_m:.2f}m<extra></extra>",
                        )
                    )

    # --- 4. Create the Dropdown Menu ---
    buttons = []
    for i, (status_name, status_col) in enumerate(status_cols.items()):
        # Create a visibility mask for the traces
        # Each status type has 6 traces (3 poses x 2 outcomes)
        visibility = [False] * len(fig.data)
        start_index = i * 6
        for j in range(6):
            if (start_index + j) < len(visibility):
                visibility[start_index + j] = True

        buttons.append(
            dict(label=status_name, method="update", args=[{"visible": visibility}])
        )

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        legend_title="Legend",
    )

    # --- 5. Save to File ---
    output_filename = "benchmark_visualization.html"
    try:
        fig.write_html(output_filename)
        print(
            f"\nInteractive visualization saved to: {os.path.abspath(output_filename)}"
        )
    except Exception as e:
        print(f"\nError saving visualization file: {e}")


def analyze_benchmark_results(file_path: str):
    """
    Loads benchmark data from a .jsonl file, calculates key metrics, prints a summary report,
    and generates interactive visualizations.
    """
    try:
        # Load the .jsonl file directly into a main DataFrame
        main_df = pd.read_json(file_path, lines=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading or parsing file: {e}")
        return

    if main_df.empty:
        print("No data found in the benchmark file.")
        return

    # --- Properly flatten the nested data into two clean DataFrames ---

    # 1. Create the trials DataFrame by flattening 'scene_characteristics'
    characteristics_df = pd.json_normalize(main_df["scene_characteristics"])
    df_trials = pd.concat(
        [
            main_df.drop(
                columns=[
                    "scene_characteristics",
                    "stages",
                    "scene_poses",
                    "parameters",
                ],
                errors="ignore",
            ),
            characteristics_df,
        ],
        axis=1,
    )

    # --- Print High-Level Summary ---
    num_trials = len(df_trials)
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

    # 2. Create the stages DataFrame by "exploding" the list of stages
    if "stages" in main_df.columns and not main_df["stages"].isnull().all():
        df_stages = (
            main_df[["trial_id", "stages"]].explode("stages").reset_index(drop=True)
        )
        # Filter out rows where stages might be null after exploding
        df_stages.dropna(subset=["stages"], inplace=True)
        stage_details = pd.json_normalize(df_stages["stages"])
        df_stages = pd.concat(
            [df_stages.drop(columns=["stages"]), stage_details], axis=1
        )

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

    # --- Generate Visualizations ---
    # Call the new function to generate and save the plots
    generate_visualizations(df_trials, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze end-to-end benchmark results and generate visualizations."
    )
    parser.add_argument(
        "benchmark_file", type=str, help="Path to the benchmark JSONL output file."
    )
    args = parser.parse_args()

    analyze_benchmark_results(args.benchmark_file)
