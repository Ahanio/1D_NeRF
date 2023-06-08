#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.functional import conv1d
import torch.nn.functional as F
from tqdm import tqdm

from data import rays_generator, unisurf_data_gen
from utils import PEncoding
from utils import to_numpy
from vis import Visualizer

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


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
        self.geom_type = "density"
        self.name = "NeRF"

    def forward(self, x):
        t = self.pe(x)
        out = self.net(t)
        # out = trunc_exp(out)
        return out

    @staticmethod
    def transmittance(density, delta, **kw):
        return torch.exp(torch.cumsum(-density * delta, dim=1))

    @staticmethod
    def surface(transmittance, density, delta):
        w = transmittance * (1 - torch.exp(-density * delta))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-5)
        return w

    def get_density(self, coords):
        return self.forward(coords)
    
    def generate_data(self, x_range, objects, N_rays, N_samples):
        return rays_generator(
            x_range, objects, N_rays, N_samples, non_uniform=True
        )


class Unisurf(nn.Module):
    step_size = 0.01

    def __init__(self, hidden_dim=32, num_freq=4, max_freq_log2=10):
        super().__init__()
        self.pe = PEncoding(num_freq=num_freq, max_freq_log2=max_freq_log2)
        self.net = nn.Sequential(
            nn.Linear(self.pe.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.name = "Unisurf"
        self.geom_type = "occupancy"

    def forward(self, x):
        t = self.pe(x).float()
        out = self.net(t)
        out = torch.sigmoid(10 * out)
        return out

    @staticmethod
    def transmittance(occ, delta, **kw):
        # return torch.exp(torch.cumsum(-density * delta, dim=1))
        return torch.cumprod(1 - occ, dim=1)

    @staticmethod
    def surface(transmittance, occ, delta):
        w = transmittance * occ
        w = w / (w.sum(dim=1, keepdim=True) + 1e-5)
        return w

    def get_density(self, coords):
        return self.forward(coords)
    
    def generate_data(self, x_range, objects, N_rays, N_samples):
        return unisurf_data_gen(
            x_range, objects, N_rays, N_samples, step_size=self.step_size
        )


def main():
    global objects

    x_range = (0, 10)
    objects = torch.Tensor([[2, 4], [6, 8]])
    N_rays = 100
    N_samples = 200
    step_size = 0.1

    model = Unisurf()  # Unisurf(step_size) # VanillaNeRF
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    visualizer = Visualizer(
        x_range, objects, enable=True, model=model
    )

    for i in tqdm(range(2000)):
        if model.name == "Unisurf":
            model.step_size = np.clip(0.1 * np.exp(-i / 1000), a_max=0.1, a_min=0.001)

        origins, x, delta, depths, first_surface = model.generate_data(
            x_range, objects, N_rays, N_samples
        )

        geom = model(x.reshape(-1, 1)).view(N_rays, N_samples)
        tr = model.transmittance(geom, delta)
        w = model.surface(tr, geom, delta)
        local_x = torch.abs(x - origins)
        pred = torch.sum(w * local_x, dim=1)

        optimizer.zero_grad()
        loss = torch.mean((pred - depths) ** 2)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Loss: {loss.item()}")
            print(f"First obj", pred[0].item() + origins[0].item())
            print(f"GT first obj", first_surface[0].item())
            print(f"Max geom value", geom[0].max().item())
            print(f"W sum", w[0].sum().item())
            print(f"Step size", step_size)

            visualizer.clear_plots()
            ray_id = 0
            visualizer.plot(x[ray_id], geom[ray_id], tr[ray_id], w[ray_id])
            visualizer.show()

    visualizer.show(block=True, interactive=True)


if __name__ == "__main__":
    main()

# %%
