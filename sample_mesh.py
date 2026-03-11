import os
import argparse
import numpy as np
import torch
from skimage import measure # type: ignore
from model import SDFModel
import time


def make_grid(res, bound, device):
    lin = torch.linspace(-bound, bound, steps=res, device=device)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1)  # (res,res,res,3)
    return pts


def write_ply(path, verts, faces):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="../runs/sdf/best.pt")
    ap.add_argument("--shape_idx", type=int, default=0)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--outdir", type=str, default="../runs/sdf/samples")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    latent_dim = ckpt["args"]["latent_dim"]
    hidden_dim = ckpt["args"]["hidden_dim"]

    model = SDFModel(latent_dim, hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load latent table
    latents = torch.nn.Embedding(
        ckpt["latents"]["weight"].shape[0],
        latent_dim
    ).to(device)

    latents.load_state_dict(ckpt["latents"])
    latents.eval()

    z = latents(torch.tensor([args.shape_idx], device=device))

    grid = make_grid(args.res, args.bound, device)
    pts = grid.reshape(-1, 3)

    sdf_vals = []

    with torch.no_grad():
        chunk = 200000
        for i in range(0, pts.shape[0], chunk):
            p = pts[i:i+chunk].unsqueeze(0)
            pred = model(p, z).squeeze(0)
            sdf_vals.append(pred.cpu().numpy())

    sdf = np.concatenate(sdf_vals).reshape(args.res, args.res, args.res)

    spacing = (2 * args.bound) / (args.res - 1)

    verts, faces, normals, _ = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=(spacing, spacing, spacing)
    )

    verts = verts - args.bound
    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.outdir, f"mesh_idx{args.shape_idx}_res{args.res}_{stamp}.ply")
    write_ply(out_path, verts, faces)
    print("mesh saved to:", out_path)


if __name__ == "__main__":
    main()