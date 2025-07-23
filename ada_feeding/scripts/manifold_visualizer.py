#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script visualizes the high-dimensional feasibility manifold data generated
by manifold_explorer.py. It uses t-SNE for dimensionality reduction to project
the 6D joint space data into a 2D plot.

The resulting interactive plot helps in understanding the structure, size, and
fragmentation of the feasible configuration space.
"""

import argparse
import sys
import os
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px


def visualize_manifold(csv_file_path: str, output_dir: str):
    """
    Loads manifold data, performs t-SNE, and generates an interactive plot.
    """
    print(f"Loading data from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_file_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Separate features (joint values) and labels (feasibility)
    features = df.drop(columns=["is_feasible"])
    labels = df["is_feasible"]

    # --- Basic Analysis: Feasibility Ratio ---
    feasibility_ratio = labels.mean()
    print(f"\nFeasibility Ratio: {feasibility_ratio:.2%}")
    print(f"Out of {len(df)} samples, {int(labels.sum())} were feasible.")

    # --- Dimensionality Reduction with t-SNE ---
    # Note: t-SNE can be slow on large datasets. For very large files (>50k samples),
    # you might consider using a subset of the data for faster iteration.
    print("\nPerforming t-SNE for dimensionality reduction (this may take a while)...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
    features_2d = tsne.fit_transform(features)
    print("t-SNE complete.")

    # --- Create a new, combined DataFrame for plotting ---
    # This DataFrame will contain the t-SNE components for the plot axes,
    # the original joint values for the hover tooltip, and the feasibility label for color.
    plot_df = pd.DataFrame(features_2d, columns=["tsne_1", "tsne_2"])

    # Reset indices of original data to ensure correct concatenation
    features.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    plot_df = pd.concat([plot_df, features, labels], axis=1)
    plot_df["is_feasible"] = plot_df["is_feasible"].astype(
        bool
    )  # Convert 0/1 to False/True for a better legend

    # --- Generate the Interactive Plot ---
    print("Generating interactive plot...")
    fig = px.scatter(
        plot_df,  # Use the new, combined DataFrame
        x="tsne_1",
        y="tsne_2",
        color="is_feasible",
        color_discrete_map={
            True: "rgba(44, 160, 44, 0.7)",  # Green, slightly transparent
            False: "rgba(214, 39, 40, 0.7)",  # Red, slightly transparent
        },
        title="2D Visualization of the Articutool Leveling Feasibility Manifold",
        labels={"color": "Is Feasible"},
        # The hover_data argument now works correctly because the 'j1', 'j2', etc.
        # columns exist in the `plot_df` DataFrame.
        hover_data={
            "tsne_1": False,  # Hide the t-SNE coordinates from the hover tooltip
            "tsne_2": False,
            "is_feasible": True,
            "j1": ":.3f",
            "j2": ":.3f",
            "j3": ":.3f",
            "j4": ":.3f",
            "j5": ":.3f",
            "j6": ":.3f",
        },
    )

    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title_text="Feasibility",
        plot_bgcolor="rgba(240, 240, 240, 0.95)",
    )

    # Save the plot to an HTML file
    output_filename = os.path.join(
        output_dir,
        f"manifold_visualization_{os.path.basename(csv_file_path).replace('.csv', '.html')}",
    )
    fig.write_html(output_filename)
    print(f"\nSuccessfully saved visualization to:\n{output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize high-dimensional manifold data using t-SNE."
    )
    parser.add_argument(
        "csv_file", type=str, help="Path to the manifold data CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="manifold_data",
        help="Directory to save the output HTML file.",
    )
    args = parser.parse_args()

    visualize_manifold(args.csv_file, args.output_dir)


if __name__ == "__main__":
    main()
