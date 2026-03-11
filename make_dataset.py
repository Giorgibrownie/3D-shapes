import os
import argparse
import shutil
import numpy as np
from tqdm import tqdm

from procedural_sdf import sample_params, sample_points, compute_sdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/superellipsoid")
    ap.add_argument("--num_shapes", type=int, default=50)
    ap.add_argument("--points_per_shape", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # If folder exists and overwrite is requested, delete it first
    if os.path.exists(args.out) and args.overwrite:
        shutil.rmtree(args.out)

    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for sid in tqdm(range(args.num_shapes), desc="Generating shapes"):
        params = sample_params(rng)
        pts = sample_points(rng, args.points_per_shape)
        sdf = compute_sdf(pts, params)

        out_path = os.path.join(args.out, f"shape_{sid:05d}.npz")
        np.savez_compressed(
            out_path,
            pts=pts,
            sdf=sdf,
            params=params,
            sid=np.int32(sid),
        )

    print(f"wrote {args.num_shapes} shapes to {args.out}")


if __name__ == "__main__":
    main()