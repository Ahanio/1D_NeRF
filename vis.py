import matplotlib.pyplot as plt
import numpy as np
import torch


def to_np(x):
    return x.detach().cpu().numpy()


class Visualizer:
    def __init__(self, x_range, objects, model, enable=True):
        self.x_range = x_range
        self.objects = objects
        self.lines = []
        self.enable = enable
        self.model = model

        if self.enable:
            plt.ion()
            self.init_plot()

    def init_plot(
        self,
    ):
        self.fig, self.ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

        self.ax[0].set_title("Density")
        self.ax[0].set_xlim(self.x_range)

        self.ax[1].set_title("Transmittance")
        self.ax[1].set_ylim(-0.01, 1.2)

        self.ax[2].set_title("Surface")
        self.ax[2].set_ylim(-0.01, 0.1)

        for num_plot in range(3):
            self.ax[num_plot].axvspan(
                -100, self.x_range[0] + 0.05, color="red", alpha=0.1
            )
            self.ax[num_plot].axvspan(
                self.x_range[1] - 0.05, 100, color="red", alpha=0.1
            )
            for obj in self.objects:
                self.ax[num_plot].axvspan(obj[0], obj[1], color="red", alpha=0.1)

            # [a.legend() for a in self.ax]

    def plot(self, x, density, tr, w, color="blue"):
        if self.enable:
            plot_params = {"linewidth": 1, "c": color}
            for num_ax, to_plot in enumerate([to_np(density), to_np(tr), to_np(w)]):
                if num_ax == 0:
                    x_density = torch.linspace(self.x_range[0], self.x_range[1], 300)[
                        :, None
                    ]
                    full_density = self.model(x_density)
                    a = self.ax[num_ax].plot(
                        to_np(x_density), to_np(full_density), "-o", **plot_params
                    )
                elif num_ax == 1:
                    a = self.ax[num_ax].plot(to_np(x), to_plot, "-o", **plot_params)
                elif num_ax == 2:
                    a = self.ax[num_ax].plot(to_np(x), to_plot, "-o", **plot_params)
                    self.ax[2].set_ylim(-0.01, max(0.1, to_plot.max()))
                self.lines.append(a)

    def clear_plots(self):
        if self.enable:
            for cur_ax in self.ax:
                for line in cur_ax.get_lines():
                    line.remove()

    def show(self, block=False):
        if self.enable:
            plt.ion()
            plt.show(block=block)
            plt.pause(0.05)
