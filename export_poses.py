#!/usr/bin/env python3
"""Export GelSLAM keyframe poses as (K, 4, 4) homogeneous transforms."""

import argparse
import os

import numpy as np

from gelslam.core.pose_graph import PoseGraphSolutions


def main():
    parser = argparse.ArgumentParser(
        description="Export long-horizon tracking poses from saved GelSLAM state."
    )
    parser.add_argument("-d", "--data_dir", required=True)
    parser.add_argument(
        "-m",
        "--method",
        required=True,
        choices=["gelslam_online", "gelslam_offline"],
    )
    args = parser.parse_args()

    load_dir = os.path.join(os.path.abspath(os.path.expanduser(args.data_dir)), args.method)
    solutions = PoseGraphSolutions.load(load_dir)
    poses = np.stack(
        [np.asarray(solutions[i], dtype=np.float32) for i in range(solutions.size())],
        axis=0,
    )
    out_npy = os.path.join(load_dir, "estimated_start_T_currs.npy")
    np.save(out_npy, poses)

    translations = poses[:, :3, 3]
    out_csv = os.path.join(load_dir, "estimated_translations_m.csv")
    np.savetxt(
        out_csv,
        translations,
        delimiter=",",
        header="x_m,y_m,z_m",
        comments="",
    )
    print(f"Wrote {poses.shape[0]} keyframe poses to {out_npy}")
    print(f"Wrote translations (meters) to {out_csv}")
    if poses.shape[0] > 1:
        path_len = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
        print(f"Path length: {path_len * 1000:.1f} mm")


if __name__ == "__main__":
    main()
