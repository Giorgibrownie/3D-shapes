import os
import argparse
import time
import numpy as np
import torch
from skimage import measure  # type: ignore
from model import SDFModel


def make_grid(res, bound, device):
    lin = torch.linspace(-bound, bound, steps=res, device=device)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1)
    return pts


def write_ply(path, verts, faces):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def reconstruct_mesh(model, z, res, bound, device):
    grid = make_grid(res, bound, device)
    pts = grid.reshape(-1, 3)

    sdf_vals = []

    with torch.no_grad():
        chunk = 200000
        for i in range(0, pts.shape[0], chunk):
            p = pts[i:i + chunk].unsqueeze(0)   # (1, chunk, 3)
            pred = model(p, z).squeeze(0)       # (chunk,)
            sdf_vals.append(pred.cpu().numpy())

    sdf = np.concatenate(sdf_vals).reshape(res, res, res)

    spacing = (2 * bound) / (res - 1)

    verts, faces, normals, values = measure.marching_cubes(
        sdf,
        level=0.0,
        spacing=(spacing, spacing, spacing)
    )

    verts = verts - bound
    return verts, faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="../runs/sdf/best.pt")
    ap.add_argument("--shape_a", type=int, default=0)
    ap.add_argument("--shape_b", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--outdir", type=str, default="../runs/sdf/interpolations")
    args = ap.parse_args()

    if args.steps < 2:
        raise ValueError("steps must be at least 2")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)

    latent_dim = ckpt["args"]["latent_dim"]
    hidden_dim = ckpt["args"]["hidden_dim"]

    model = SDFModel(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    latents = torch.nn.Embedding(
        ckpt["latents"]["weight"].shape[0],
        latent_dim
    ).to(device)
    latents.load_state_dict(ckpt["latents"])
    latents.eval()

    z_a = latents(torch.tensor([args.shape_a], device=device))  # (1, latent_dim)
    z_b = latents(torch.tensor([args.shape_b], device=device))  # (1, latent_dim)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    ts = np.linspace(0.0, 1.0, args.steps)

    for i, t in enumerate(ts):
        z_t = (1.0 - t) * z_a + t * z_b

        verts, faces = reconstruct_mesh(
            model=model,
            z=z_t,
            res=args.res,
            bound=args.bound,
            device=device
        )

        out_path = os.path.join(
            args.outdir,
            f"interp_A{args.shape_a}_B{args.shape_b}_step{i:02d}_t{t:.2f}_{stamp}.ply"
        )
        write_ply(out_path, verts, faces)
        print(f"saved {out_path}")

    print("done.")


if __name__ == "__main__":
    main()