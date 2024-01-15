#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for modifying the acquisition library
"""

import numpy as np
import os
import sys
import yaml


def main(in_fname: str, out_fname: str):
    target_rads = 0.3
    rad_thresh = 0.1

    # Load YAML
    print(f"Reading from: {in_fname}")
    with open(in_fname, "r") as f:
        data = yaml.safe_load(f)
    actions = data["actions"]

    # Scale Speeds
    for action in actions:
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
        print(f"Warning, overwritting: {sys.argv[2]}")
    main(sys.argv[1], sys.argv[2])
