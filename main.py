#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from vis import Visualizer
from utils import to_numpy
from models import PEncoding


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


def rays_generator(x_range, objects, num_rays, num_samples_per_ray):
    x = torch.linspace(x_range[0], x_range[1], num_rays * 10).float().unsqueeze(-1)

    # pick random points that lay outside of objects
    x = x[torch.randperm(x.shape[0])]
    for ob in objects:
        x = x[torch.logical_not(torch.logical_and(x > ob[0], x < ob[1]))]
    x = x[:num_rays]
    origins = x  # [n,]
    origins[0] = 0.2  ## For visualization purposes

    directions = torch.Tensor(np.random.choice([-1, 1], num_rays)).float()  # [n,]
    directions[0] = 1  ## For visualization purposes

    t_max = torch.zeros_like(origins)
    t_max[directions > 0] = x_range[1] - origins[directions > 0] + 1
    t_max[directions < 0] = origins[directions < 0] - x_range[0] + 1
    t_min = 0

    ray_points = origins.unsqueeze(-1) + directions.unsqueeze(-1) * (
        torch.linspace(0, 1, num_samples_per_ray) * (t_max - t_min).unsqueeze(-1)
        + t_min
    )

    # for each ray find the depth along that ray where it intersects with the object.
    # If the ray does not intersect with any object, then the depth is the max depth
    depths = []
    bounds = torch.Tensor(objects).reshape(-1).sort()[0]
    for i in range(num_rays):
        cur_depth = None
        if directions[i] > 0:
            closest = bounds[bounds > origins[i]]
            if len(closest) != 0:
                cur_depth = closest.min() - origins[i]
            else:
                cur_depth = x_range[1] - origins[i]
        elif directions[i] < 0:
            closest = bounds[bounds < origins[i]]
            if len(closest) != 0:
                cur_depth = origins[i] - closest.max()
            else:
                cur_depth = origins[i] - x_range[0]

        depths.append(cur_depth)

    depths = torch.Tensor(depths)
    # return origins, directions, depths
    gt_boundary = origins + directions * depths
    delta = torch.abs(
        torch.diff(ray_points, dim=1, append=ray_points[:, 0:1])
    )  # prepend=ray_points[:,0:1]
    # ray_points = ray_points[:, :-1]
    return origins.unsqueeze(-1), ray_points, delta, depths, gt_boundary


def main():
    x_range = (0, 10)
    objects = torch.Tensor([[3, 5], [7, 8]])
    N_rays = 100
    N_samples = 300

    model = VanillaNeRF()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
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

    visualizer.show(block=True)


if __name__ == "__main__":
    main()
