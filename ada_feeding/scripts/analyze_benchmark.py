import argparse
import json
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_visualizations(df_trials: pd.DataFrame, file_path: str):
    """
    Generates interactive Plotly visualizations with a context-aware dropdown
    for stage-specific analysis.
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
    # Fill NaN for trials that didn't reach certain stages
    df_stage_status = df_stage_status.fillna(0)
    df_stage_status.columns = [f"status_{col}" for col in df_stage_status.columns]

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

    # --- 3. Create Figure and Six Persistent Traces ---
    fig = go.Figure()
    symbols = {"food_pose": "circle", "mouth_pose": "square", "resting_pose": "diamond"}
    colors = {"Success": "green", "Failure": "red"}

    # Define the default view
    default_status_name = "End-to-End"
    default_status_col = "end_to_end_success"
    relevance_map = {
        "End-to-End": ["food_pose", "mouth_pose", "resting_pose"],
        "Stage: HomeToAbovePlate": ["food_pose"],
        "Stage: AbovePlateToAboveFood": ["food_pose"],
        "Stage: AboveFoodToInFood": ["food_pose"],
        "Stage: LevelArticutool": ["food_pose"],
        "Stage: Resting": ["food_pose", "resting_pose"],
    }

    # Create the 6 persistent traces, populating them with data for the default view
    for pose_name, symbol in symbols.items():
        for outcome, color in colors.items():
            success_val = 1 if outcome == "Success" else 0

            # Filter data for this trace's default view
            df_subset = vis_df[
                (vis_df["pose_name"] == pose_name)
                & (vis_df[default_status_col] == success_val)
            ]

            x_data, y_data, z_data, custom_data, hover_template = (
                [],
                [],
                [],
                None,
                "none",
            )
            if pose_name in relevance_map.get(default_status_name, []):
                x_data = df_subset["x"]
                y_data = df_subset["y"]
                z_data = df_subset["z"]
                custom_data = df_subset.to_dict("records")
                hover_template = (
                    "<b>Trial ID: %{customdata.trial_id}</b><br>Pose: %{customdata.pose_name}<br>Status: "
                    + outcome
                    + "<br>Distance: %{customdata.food_mouth_distance_m:.2f}m<extra></extra>"
                )

            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    customdata=custom_data,
                    hovertemplate=hover_template,
                    mode="markers",
                    marker=dict(color=color, symbol=symbol, size=5, opacity=0.7),
                    name=f"{outcome} ({pose_name.replace('_pose', '')})",
                )
            )

    # --- 4. Create the Dropdown Menu with 'restyle' Logic ---
    status_cols = {"End-to-End": "end_to_end_success"}
    stage_names = sorted([c.replace("status_", "") for c in df_stage_status.columns])
    for stage_name in stage_names:
        status_cols[f"Stage: {stage_name}"] = f"status_{stage_name}"
    buttons = []

    # Create all traces upfront
    for status_name, status_col in status_cols.items():
        update_args = {"x": [], "y": [], "z": [], "customdata": [], "hovertemplate": []}

        for pose_name in symbols.keys():
            for outcome in ["Success", "Failure"]:
                success_val = 1 if outcome == "Success" else 0
                df_subset = vis_df[
                    (vis_df["pose_name"] == pose_name)
                    & (vis_df[status_col] == success_val)
                ]

                if pose_name in relevance_map.get(status_name, []):
                    update_args["x"].append(df_subset["x"])
                    update_args["y"].append(df_subset["y"])
                    update_args["z"].append(df_subset["z"])
                    update_args["customdata"].append(df_subset.to_dict("records"))
                    update_args["hovertemplate"].append(
                        "<b>Trial ID: %{customdata.trial_id}</b><br>Pose: %{customdata.pose_name}<br>Status: "
                        + outcome
                        + "<br>Distance: %{customdata.food_mouth_distance_m:.2f}m<extra></extra>"
                    )
                else:
                    for key in ["x", "y", "z", "customdata", "hovertemplate"]:
                        update_args[key].append(None)  # Use None to clear data

        buttons.append(dict(label=status_name, method="restyle", args=[update_args]))

    fig.update_layout(
        title="3D Workspace Visualization",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
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
