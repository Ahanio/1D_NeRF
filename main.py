#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.functional import conv1d
from tqdm import tqdm

from data import rays_generator
from models import PEncoding
from utils import to_numpy
from vis import Visualizer


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
        self.fix_forward = True

    def forward(self, x):
        if self.fix_forward:
            out = torch.logical_and(x > objects[0][0], x < objects[0][1])
            border = torch.logical_not(torch.logical_and(x > 0, x < 10))
            out = torch.logical_or(out, border).float()
        else:
            t = self.pe(x)
            out = self.net(t)  # torch.clamp(self.net(t), max=1)
        return out

    @staticmethod
    def transmittance(density, delta):
        # modified_occ = torch.cumprod((1 - density) ** (delta), dim=1)
        # modified_occ = torch.exp(
        #     torch.cumsum(torch.log(1 - density + 1e-5) * delta, dim=1)
        # )
        old_density = -torch.log(1 - density + 1e-5)
        modified_occ = torch.exp(torch.cumsum(-old_density * delta, dim=1))

        # kern = torch.Tensor([1, 2, 1]) / 4
        # # kern = torch.Tensor([1, 6, 15, 20, 15, 6, 1]) / 64
        # conv_modified_occ = conv1d(
        #     modified_occ[:, None],
        #     kern[None, None, :],
        #     padding="same",
        # )[:, 0]
        # conv_modified_occ[:, 0] = conv_modified_occ[:, 1]
        # conv_modified_occ[:, -1] = conv_modified_occ[:, -2]

        conv_modified_occ = modified_occ

        # torch.exp(torch.cumsum(-density * delta, dim=1))
        return conv_modified_occ

    @staticmethod
    def surface(transmittance, density, delta):
        # return transmittance * (1 - torch.exp(-density * delta))
        # return transmittance * density  # * delta
        return -torch.diff(transmittance, dim=1, append=transmittance[:, -2:-1])


class UNISURF(VanillaNeRF):
    def __init__(self, hidden_dim=32, num_freq=4, max_freq_log2=10):
        super().__init__(hidden_dim, num_freq, max_freq_log2)
        self.net = nn.Sequential(
            nn.Linear(self.pe.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def transmittance(density, delta):
        # transm = torch.exp(torch.cumsum(-old_density * delta, dim=1))
        occ = density
        transm = torch.cumprod((1 - occ), dim=1)
        return transm

    @staticmethod
    def surface(transmittance, density, delta):
        # return transmittance * (1 - torch.exp(-density * delta))
        # return transmittance * density  # * delta
        occ = density
        return transmittance * occ


def main():
    x_range = (0, 10)
    global objects
    objects = torch.Tensor([[3, 5]])  # , [8, 9]])
    N_rays = 100
    N_samples = 200

    model = OccRF()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    visualizer = Visualizer(x_range, objects, enable=True, model=model)
    visualizer.show(block=True, interactive=True)

    for i in tqdm(range(3000)):
        origins, x, delta, depths, first_surface = rays_generator(
            x_range, objects, N_rays, N_samples, non_uniform=True
        )

        density = model(x.reshape(-1, 1)).view(N_rays, N_samples)
        tr = model.transmittance(density, delta)
        w = model.surface(tr, density, delta)
        local_x = torch.abs(x - origins)
        pred = torch.sum(w * local_x, dim=1)

        reg_coords = torch.rand(1, 100) * 10
        shift = torch.randn(1, 100) * 0.01

        reg_pred = model(reg_coords.reshape(-1, 1))
        shift_reg_pred = model((reg_coords + shift).reshape(-1, 1))
        reg_loss = torch.mean(torch.pow(reg_pred - shift_reg_pred, 2))

        optimizer.zero_grad()
        loss = torch.mean((pred - depths) ** 2) + reg_loss
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Loss: {loss.item()}")
            print(f"First obj", pred[0].item() + origins[0].item())
            print(f"GT first obj", first_surface[0].item())
            print(f"W sum", w[0].sum().item())

            visualizer.clear_plots()
            ray_id = 0
            visualizer.plot(x[ray_id], density[ray_id], tr[ray_id], w[ray_id])
            visualizer.show()

    visualizer.show(block=True, interactive=True)


if __name__ == "__main__":
    main()
