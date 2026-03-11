import os
import argparse
import random
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 

from dataset import SDFDataset
from model import SDFModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/superellipsoid")
    ap.add_argument("--out", type=str, default="../runs/sdf")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_shapes", type=int, default=4)
    ap.add_argument("--points_batch", type=int, default=2048)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("device:", device)

    ds = SDFDataset(args.data)
    num_shapes = len(ds)

    dl = DataLoader(
        ds,
        batch_size=args.batch_shapes,
        shuffle=True,
        num_workers=0,
    )

    model = SDFModel(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    latents = nn.Embedding(num_shapes, args.latent_dim).to(device)
    nn.init.normal_(latents.weight, mean=0.0, std=0.01)

    opt = torch.optim.Adam(
        list(model.parameters()) + list(latents.parameters()),
        lr=args.lr
    )

    loss_fn = nn.L1Loss()

    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        nsteps = 0

        for batch in dl:
            pts = batch["pts"].to(device)   # (B, N, 3)
            sdf = batch["sdf"].to(device)   # (B, N)
            idx = batch["idx"].to(device)   # (B,)

            B, N, _ = pts.shape

            j = torch.randint(0, N, (B, args.points_batch), device=device)

            pts_s = torch.gather(
                pts,
                1,
                j.unsqueeze(-1).expand(-1, -1, 3)
            )  # (B, P, 3)

            sdf_s = torch.gather(
                sdf,
                1,
                j
            )  # (B, P)

            # Get the latent vector for each shape in the batch
            z = latents(idx)   

            # Predict SDF
            pred = model(pts_s, z)   # (B, P)

            # Compute loss
            loss = loss_fn(pred, sdf_s)

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            nsteps += 1

        avg = total / max(nsteps, 1)
        print(f"epoch {epoch:03d} | train L1: {avg:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "latents": latents.state_dict(),
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.out, "last.pt"))

        if avg < best:
            best = avg
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print("  saved best.pt")

    print("done. best loss:", best)


if __name__ == "__main__":
    main()