import torch # type: ignore
import torch.nn as nn # type: ignore

class SDFModel(nn.Module):

    def __init__(self, latent_dim=16, hidden_dim=128):
        super().__init__()

        input_dim = 3 + latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, xyz, latent):
        B, N, _ = xyz.shape

        latent_expanded = latent.unsqueeze(1).expand(-1, N, -1)

        x = torch.cat([xyz, latent_expanded], dim=-1)

        x = x.view(B * N, -1)

        out = self.net(x)

        out = out.view(B, N)

        return out