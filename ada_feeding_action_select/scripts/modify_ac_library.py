#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for modifying the acquisition library
"""

import math
import os
import sys

import numpy as np
import yaml


def main(in_fname: str, out_fname: str):
    """
    Modify acquisition library
    """

    # pylint: disable=too-many-locals

    # Scale Angular Speeds
    target_rads = 0.3

    # Clear low-angle motions
    rad_thresh = 0.1

    # Max pre_offset x/y based on pre_pose
    max_offset = 0.01

    # Target below top of food
    food_depth = -0.02

    # Load YAML
    print(f"Reading from: {in_fname}")
    with open(in_fname, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    actions = data["actions"]

    for action in actions:
        # Add up to max_offset to pre_offset
        norm = math.sqrt(
            action["pre_pos"][0] * action["pre_pos"][0]
            + action["pre_pos"][1] * action["pre_pos"][1]
        )
        if not np.isclose(norm, 0.0):
            scale = min(max_offset, norm) / norm
            action["pre_offset"][0] = scale * action["pre_pos"][0]
            action["pre_offset"][1] = scale * action["pre_pos"][1]

        # Target below top of food
        action["pre_offset"][2] = food_depth

        # Scale Angular Speeds and clear low-angle motions
        for prefix in ["grasp", "ext"]:
            tot = (
                np.linalg.norm(np.array(action[f"{prefix}_angular"]))
                * action[f"{prefix}_duration"]
            )
            if tot < rad_thresh:
                action[f"{prefix}_angular"] = [0.0, 0.0, 0.0]
                continue

            speed_ratio = float(
                target_rads / np.linalg.norm(np.array(action[f"{prefix}_angular"]))
            )
            action[f"{prefix}_angular"] = [
                float(val * speed_ratio) for val in action[f"{prefix}_angular"]
            ]
            action[f"{prefix}_linear"] = [
                float(val * speed_ratio) for val in action[f"{prefix}_linear"]
            ]
            action[f"{prefix}_duration"] /= speed_ratio

            new_tot = (
                np.linalg.norm(np.array(action[f"{prefix}_angular"]))
                * action[f"{prefix}_duration"]
            )
            assert np.isclose(tot, new_tot)

    # Write new stuff
    print(f"Writing to: {out_fname}")
    data["actions"] = actions
    with open(out_fname, "w", encoding="utf-8") as file:
        file.write("# This file was modifed by modify_ac_library.py\n")
        yaml.dump(data, file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 modify_ac_library.py in_path out_path")
        sys.exit(1)
    if not os.path.isfile(sys.argv[1]):
        print(f"File Not Found: {sys.argv[1]}")
        sys.exit(1)
    if os.path.isfile(sys.argv[2]):
        print(f"Warning, overwriting: {sys.argv[2]}")
    main(sys.argv[1], sys.argv[2])
