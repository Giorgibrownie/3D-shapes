import os
import glob
import numpy as np
import matplotlib.pyplot as plt # type: ignore


def plot_shape(npz_path, threshold=0.10):
    data = np.load(npz_path)
    pts = data["pts"]
    sdf = data["sdf"]

    surface_pts = pts[np.abs(sdf) < threshold]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        surface_pts[:, 0],
        surface_pts[:, 1],
        surface_pts[:, 2],
        s=10
    )

    ax.set_title(os.path.basename(npz_path))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_four_shapes(folder, threshold=0.02):
    files = sorted(glob.glob(os.path.join(folder, "shape_*.npz")))[:4]

    fig = plt.figure(figsize=(12, 12))

    for i, path in enumerate(files, 1):
        data = np.load(path)
        pts = data["pts"]
        sdf = data["sdf"]

        surface_pts = pts[np.abs(sdf) < threshold]

        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(
            surface_pts[:, 0],
            surface_pts[:, 1],
            surface_pts[:, 2],
            s=2
        )
        ax.set_title(os.path.basename(path))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_shape("data/superellipsoid/shape_00000.npz")