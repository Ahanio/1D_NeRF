#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import conv1d

from vis import Visualizer
from utils import to_numpy
from models import PEncoding
from data import rays_generator


class VanillaNeRF(nn.Module):
    def __init__(self, hidden_dim=32, num_freq=4, max_freq_log2=10):
        super().__init__()
        self.pe = PEncoding(num_freq=num_freq, max_freq_log2=max_freq_log2)
        self.net = nn.Sequential(
            nn.Linear(self.pe.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        t = self.pe(x)
        return self.net(t)

    @staticmethod
    def transmittance(density, delta):
        return torch.exp(torch.cumsum(-density * delta, dim=1))

    @staticmethod
    def surface(transmittance, density, delta):
        return transmittance * (1 - torch.exp(-density * delta))


class OccRF(VanillaNeRF):
    def __init__(self, hidden_dim=32, num_freq=4, max_freq_log2=10):
        super().__init__(hidden_dim, num_freq, max_freq_log2)
        self.net = nn.Sequential(
            nn.Linear(self.pe.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # nn.Sigmoid(),
        )

    def forward(self, x):
        t = self.pe(x)
        out = self.net(t)  # torch.clamp(self.net(t), max=1)
        return out

    @staticmethod
    def transmittance(density, delta):
        modified_occ = torch.cumprod(1 - density * delta, dim=1)

        kern = torch.Tensor([1, 2, 1]) / 4
        # kern = torch.Tensor([1, 6, 15, 20, 15, 6, 1]) / 64
        conv_modified_occ = conv1d(
            modified_occ[:, None],
            kern[None, None, :],
            padding="same",
        )[:, 0]
        # conv_modified_occ[:, 0] = conv_modified_occ[:, 1]
        # conv_modified_occ[:, -1] = conv_modified_occ[:, -2]

        # conv_modified_occ = modified_occ

        # torch.exp(torch.cumsum(-density * delta, dim=1))
        return conv_modified_occ

    @staticmethod
    def surface(transmittance, density, delta):
        # return transmittance * (1 - torch.exp(-density * delta))
        return -torch.diff(transmittance, dim=1, append=transmittance[:, -2:-1])


def main():
    x_range = (0, 10)
    objects = torch.Tensor([[3, 5]])  # , [8, 9]])
    N_rays = 100
    N_samples = 300

    model = OccRF()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    visualizer = Visualizer(x_range, objects, enable=True, model=model)

    for i in tqdm(range(5000)):
        origins, x, delta, depths, first_surface = rays_generator(
            x_range, objects, N_rays, N_samples
        )

        density = model(x.reshape(-1, 1)).view(N_rays, N_samples)
        tr = model.transmittance(density, delta)
        w = model.surface(tr, density, delta)
        local_x = torch.abs(x - origins)
        pred = torch.sum(w * local_x, dim=1)

        optimizer.zero_grad()
        # mask = torch.logical_and(first_surface != 10, first_surface != 0)
        loss = torch.mean((pred - depths) ** 2)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Loss: {loss.item()}")
            print(f"First obj", pred[0].item() + origins[0].item())
            print(f"W sum", w[0].sum().item())

            visualizer.clear_plots()
            ray_id = 0
            visualizer.plot(x[ray_id], density[ray_id], tr[ray_id], w[ray_id])
            visualizer.show()

    visualizer.show(block=True, interactive=True)


if __name__ == "__main__":
    main()
