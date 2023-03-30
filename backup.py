import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from vis import Visualizer
from utils import to_numpy


class VanillaNeRF(nn.Module):
    def __init__(self, hidden_dim=32, pe_order=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )
        self.pe_order = pe_order

    def forward(self, x):
        tlist = []
        for i in range(self.pe_order):
            tlist.append(torch.sin(x * i))
            tlist.append(torch.cos(x * i))
        t = torch.cat(tlist, dim=1)
        return self.net(t)

    @staticmethod
    def transmittance(density, delta):
        return torch.exp(torch.cumsum(-density * delta, dim=0))

    @staticmethod
    def surface(transmittance, density, delta):
        return transmittance * (1 - torch.exp(-density * delta))


class Ray:
    def __init__(self, origin, direction, t_max, t_min=0, num_pts=100):
        self.origin = origin
        self.direction = direction
        self.num_pts = num_pts

        self.t_min = t_min
        self.t_max = t_max

        self.ray_points = self.sample_pts()

    def sample_pts(self):
        return self.origin + self.direction * torch.linspace(
            self.t_min, self.t_max, self.num_pts
        ).unsqueeze(-1)



def main():
    x_range = (0, 10)
    x = torch.linspace(x_range[0], x_range[1], 100).float().unsqueeze(-1)
    delta = torch.diff(x, prepend=x[0:1], dim=0)

    objects = torch.Tensor([[3, 5]])

    visualizer = Visualizer(x_range, objects, x)

    model = VanillaNeRF()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    first_surface = torch.Tensor([min([i[0] for i in objects])]).float()
    for i in range(200):
        density_first = model(x)
        tr_first = model.transmittance(density_first, delta)
        w_first = model.surface(tr_first, density_first, delta)
        pred_first = torch.sum(w_first * x, dim=0)

        optimizer.zero_grad()
        loss_depth = (pred_first - first_surface) ** 2
        loss_density = (density_first[x[:, 0] < (first_surface - 1)] ** 2).sum()
        loss = loss_depth + loss_density
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Loss: {loss.item()}")
            print(f"Pred first: {pred_first.item()}")

            visualizer.clear_plots()
            visualizer.plot(
                to_numpy(density_first),
                to_numpy(tr_first),
                to_numpy(w_first) / (to_numpy(w_first).sum() + 1e-5),
            )
            visualizer.show()

    visualizer.show(block=True)


if __name__ == "__main__":
    main()
